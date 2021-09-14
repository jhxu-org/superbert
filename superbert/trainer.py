from superbert.utils.misc import get_rank
import torch
from superbert.utils.my_metrics import AbeAccuracy
from torch.nn.parallel.distributed import (DistributedDataParallel,
                                           _find_tensors)
import logging
from torch.nn.parallel import DataParallel
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from superbert.datasets.samplers.distributed_sampler import DistributedSampler
import os
import copy, time, json
import datetime
from tqdm import tqdm
import sys
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
)
handler.setFormatter(formatter)
root.addHandler(handler)
logger = logging.getLogger(os.path.basename(__name__))
log_json = []
def train(args, train_dataset, eval_dataset, model, meters):
    """ Train the model """
    #if args.local_rank in [-1, 0]: tb_writer = SummaryWriter()

    args.train_batch_size = args.samples_per_gpu * max(1, args.n_gpu)
    train_sampler = None if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset, num_workers=args.workers_per_gpu, pin_memory=False, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.datasets[0].collate)

    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total) # original

    if args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    if args.resume_path != None:  # recovery
        logger.info(
            "Load BERT optimizer from {}".format(args.resume_path))
        optimizer_to_load = torch.load(
            os.path.join(args.resume_path, 'optimizer.pth'),
            map_location=torch.device("cpu"))
        optimizer.load_state_dict(optimizer_to_load.pop("optimizer"))
        scheduler.load_state_dict(optimizer_to_load.pop("scheduler"))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model).cuda()

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    if args.local_rank != -1:
        logger.info("***** barrier *****")
        torch.distributed.barrier()

    # Train!
    if get_rank() == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.samples_per_gpu)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    #train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    best_score = 0
    best_model = {
        'epoch': 0,
        'model': copy.deepcopy(model.state_dict()), #model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    #eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)

    for epoch in range(int(args.num_train_epochs)):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0

        # if args.adjust_dp and epoch>=3:
        #     logger.info("change droput ratio {} to 0.3".format(args.drop_out))
        #     if hasattr(model, 'module'):
        #         model.module.dropout.p = 0.3
        #         model.module.bert.dropout.p = 0.3
        #         model.module.bert.embeddings.dropout.p = 0.3
        #     else:
        #         model.dropout.p = 0.3
        #         model.bert.dropout.p = 0.3
        #         model.bert.embeddings.dropout.p = 0.3

        # if args.adjust_loss and epoch>=args.adjust_loss_epoch:
        #     logger.info("\t change loss type from kl to bce")
        #     model.loss_type = 'bce'

        # debug
        #epoch = 20
        #global_step = epoch*math.ceil(len(train_dataset)/(args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))

        t_start = time.time()
        if (args.local_rank != -1):
            torch.distributed.barrier()
        max_iter = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            # if step >= 100:
            #     break
            model.train()
            start1 = time.time()
            outputs = model(batch)
            loss = sum([v for k,v in outputs.items() if "loss" in k])
            #loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            # loss, logits = outputs[:2]

            #loss = instance_bce_with_logits(logits, batch[4])

            # if args.n_gpu > 1: loss = loss.mean() # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                count_norm += 1
            # batch_score
            # batch_score = compute_score_with_logits(logits, batch[4]).sum()
            # train_score += batch_score.item()
            batch_time = time.time() - start1
            tr_loss = loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # if get_rank() == 0:
                accus =  {k:v.item() for k,v in outputs.items() if "accuracy" in k}
                item_loss =  {k:v.item() for k,v in outputs.items() if "loss" in k}
                metrics_to_log = {
                            'time_info': {'compute': batch_time},
                            'batch_metrics': {'loss': tr_loss, **item_loss, **accus,}
                }
                params_to_log = {'params': {'bert_lr': optimizer.param_groups[0]["lr"]}}
                meters.update_metrics(metrics_to_log)
                meters.update_params(params_to_log)
                if args.local_rank in [-1, 0] and args.log_period > 0 and global_step % args.log_period == 0:# Log metrics
                    if args.local_rank not in [-1, 0]:
                        torch.distributed.barrier()
                    avg_time = meters.meters['time_info']['compute'].global_avg
                    eta_seconds = avg_time * (t_total - step - 1)
                    eta_string = str(
                        datetime.timedelta(seconds=int(eta_seconds)))
                    logger.info(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                "iter: {iter}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=step + 1,
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        ) + "\n    " + meters.get_logs(step + 1)
                    )
            if (step + 1) == max_iter or (step + 1) % args.ckpt_period == 0:  # Save a trained model
                # log_json[step+1] = tr_loss
            
                # reset metrics
                tr_loss = 0

                if get_rank() == 0:
                    # report metrics
                    model_to_save = model.module if hasattr(
                        model,
                        'module') else model  # Take care of distributed/parallel training
                    optimizer_to_save = {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch":epoch,
                        "step":step
                        }
                    output_dir = os.path.join(args.output_dir, 'last_checkpoint')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, "superbert.pth"))
                    torch.save(optimizer_to_save,
                            os.path.join(output_dir,
                                        'optimizer.pth'))     
                    logger.info(
                        "Saving model checkpoint {0} to {1}".format(
                            step + 1, output_dir))
                    # if args.local_rank in [-1, 0] and args.evaluate_durival_check_intervalng_training:
                    # #if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    #     logger.info("Epoch: %d, global_step: %d" % (epoch, global_step))
                    #     eval_result, eval_score, upper_bound = evaluate(args, model, eval_dataset, prefix=global_step)
                    #     if eval_score > best_score:
                    #         best_score = eval_score
                    #         best_model['epoch'] = epoch
                    #         best_model['model'] = copy.deepcopy(model)

                    #     logger.info("EVALERR: {}%".format(100 * best_score))

                    # if args.local_rank == 0:
                    #     torch.distributed.barrier()

                    # logging_loss = tr_loss

            #if args.max_steps > 0 and global_step > args.max_steps:
            #    epoch_iterator.close()
            #    break
        # if get_rank() == 0:
            # evaluation
        if (args.local_rank != -1):
            torch.distributed.barrier()
        eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
        logger.info("Epoch: %d, global_step: %d eval_score:%.04f" % (epoch, global_step, eval_score))
        if get_rank() == 0:
            if eval_score > best_score:
                best_score = eval_score
                best_model['epoch'] = epoch
                best_model['model'] = copy.deepcopy(model.state_dict())
                best_model['optimizer'] = copy.deepcopy(optimizer.state_dict())

        # save checkpoints
        if (args.local_rank in [-1, 0]) and (args.save_epoch>0 and epoch%args.save_epoch == 0) and (epoch>args.save_after_epoch):
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
            logger.info(f"in output_dir:{output_dir}")
            if not os.path.exists(output_dir): os.makedirs(output_dir)

            model_to_save = best_model['model']  # Take care of distributed/parallel training

            save_num = 0
            torch.save(model_to_save, os.path.join(output_dir, "superbert.pth"))
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            
            logger.info("Saving model checkpoint {0} to {1}".format(epoch, output_dir))
        if get_rank() == 0:
            epoch_log = {'epoch': epoch, 'eval_score': eval_score, 'best_score':best_score}
            log_json.append(epoch_log)
            if args.local_rank in [-1, 0]:
                with open(args.output_dir + f'/eval_logs.json', 'w') as fp:
                    json.dump(log_json, fp)

            logger.info("PROGRESS: {}%".format(round(100*(epoch + 1) / args.num_train_epochs, 4)))
            logger.info("EVALERR: {}%".format(100*best_score))

            t_end = time.time()
            logger.info('Epoch: %d, Train Time: %.3f' % (epoch, t_end - t_start))

        #if args.max_steps > 0 and global_step > args.max_steps:
        #    train_iterator.close()
        #    break

    if args.local_rank in [-1, 0]: # Save the final model checkpoint
        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
            json.dump(log_json, fp)

        output_dir = os.path.join(args.output_dir, 'best-{}'.format(best_model['epoch']))
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        model_to_save = best_model['model'] #.module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training
        logger.info(f"out output_dir:{output_dir}")
        save_num = 0
        # model_to_save.save_pretrained(output_dir)
        torch.save(model_to_save, os.path.join(output_dir, "superbert.pth"))
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))

        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset=None, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_outputs_dirs = args.output_dir
    print(args.output_dir)
    #if args.n_gpu > 1: model = torch.nn.DataParallel(model) # debug: single-gpu or multi-gpus

    results = []
    t_start = time.time()
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]: os.makedirs(args.output_dir)

    args.eval_batch_size = args.samples_per_gpu * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, num_workers=args.workers_per_gpu, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.datasets[0].collate)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    num_data = 0
    score = 0
    upper_bound = 0
    results_dict = {}
    size = 0
    abe_accuracy = AbeAccuracy(compute_on_step=False).cuda()
    for idx, batch in tqdm((enumerate(eval_dataloader))):
    #for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        # batch = tuple(t.to(args.device) for t in batch)
        # if idx >= 100:
        #     break
        with torch.no_grad():
            
            outputs = model(batch)

            eval_loss += sum([v for k,v in outputs.items() if "loss" in k])
            
        nb_eval_steps += 1

        #if preds is None:
        #    preds = logits.detach().cpu().numpy()
        #    out_label_ids = inputs['labels'].detach().cpu().numpy()
        #else:
        #    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        #    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    if hasattr(model, 'module'):
        score, score_dict = model.module.get_val_metric()
    else:
        score, score_dict = model.get_val_metric()
    
    logger.info("***** compute score *****")
    # score = abe_accuracy.compute().item()#len(eval_dataloader.dataset)
    # upper_bound = upper_bound / len(eval_dataloader.dataset)

    logger.info("Eval Results:")
    for k,v in score_dict.items():
        logger.info(f"{k} eval Score: %.3f" % (100*v.item()))
    # with open(os.path.join(args.data_dir, 'val_results.json'),
    #           'w') as f:
    #     json.dump(results_dict, f)

    t_end = time.time()
    logger.info('Eva Time Cost: %.3f' % (t_end - t_start))

        #eval_loss = eval_loss / nb_eval_steps
        #if args.output_mode == "classification":
        #    preds = np.argmax(preds, axis=1)
        #elif args.output_mode == "regression":
        #    preds = np.squeeze(preds)
        #result = compute_metrics(eval_task, preds, out_label_ids)
        #results.update(result)

        #output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        #with open(output_eval_file, "w") as writer:
        #    logger.info("***** Eval results {} *****".format(prefix))
        #    for key in sorted(result.keys()):
        #        logger.info("  %s = %s", key, str(result[key]))
        #        writer.write("%s = %s\n" % (key, str(result[key])))

    return score.item()