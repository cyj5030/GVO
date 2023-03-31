import torch
import numpy as np

import utils
import model
import logger

from data.datasets import load_kitti, load_kaist
import evaluation as Eval

def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_steps=10):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (decay_rate ** (epoch // decay_steps))

def trainer(net, datasets, log, cfgs): 
    dataloader = torch.utils.data.DataLoader(datasets['train'], batch_size=cfgs['batchsize'], shuffle=True, num_workers=4, drop_last=True)
    
    # parameters
    lr = cfgs['lr']
    device = cfgs['device']

    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0}

    # optim
    if cfgs['model'] == 'flow':
        params = net.parameters()
    elif 'depth' in cfgs['model']: 
        params = [
            {'params': net.depth_net.parameters()}
        ]
    elif cfgs['model'] == 'flow-ft':
        params = [
            {'params': net.flow_model.parameters()}
        ]
        # params = net.parameters()
    optim = torch.optim.Adam(params=params, lr=lr, weight_decay=0)

    # use pretrain model?
    if cfgs['reuse']:
        state, model_params = log.load(folder=cfgs['reuse_root'], device=device,checkpoints=None)
        print('Load params from %s.' % (cfgs['reuse_root']))
        if cfgs['reuse_part'] == 'all':
            net.load_state_dict(model_params)
        elif cfgs['reuse_part'] == 'flow':
            net.flow_model.load_state_dict(model_params)
            
    # star training
    net = net.to(device)
    for epoch in range(cfgs['train_epoch']):
        learning_rate_decay(optim, epoch=epoch, decay_rate=cfgs['decay_rate'], decay_steps=cfgs['decay_step'])
        state_dict['epoch'] += 1
        for data in dataloader:
            for key, itms in data.items():
                data[key] = itms.to(device)

            # calc loss and flow
            net.train()
            data['state'] = state_dict
            losses, loss = net(data)

            # update params
            optim.zero_grad()
            loss.backward()
            optim.step()

            # update state dict
            state_dict['itr'] += 1

            # print for every 10 iter
            if state_dict['itr'] % cfgs['print_iter'] == 0:
                log.print_losses(losses, state_dict, write=True)

            # eval for every 1000 iter
            if state_dict['itr'] % cfgs['eval_iter'] == 0:
                Eval.eval_kitti_flow(net=net, datasets=datasets['val'], logger=log, cfgs=cfgs, write=True)
                Eval.eval_kitti_odometry(net=net, datasets=datasets['val'], logger=log, cfgs=cfgs, kpts=None, write=True)
                Eval.eval_kitti_depth(net=net, datasets=datasets['val'], logger=log, cfgs=cfgs, write=True)
            
            #save for every 10000 iterd
            if state_dict['itr'] % cfgs['save_iter'] == 0:
                print('START SAVE AND EVAL...')
                log.save(net=net, state=state_dict, folder='model')
                Eval.eval_kitti_flow(net=net, datasets=datasets['test'], logger=log, cfgs=cfgs)
                Eval.eval_kitti_odometry(net=net, datasets=datasets['test'], logger=log, cfgs=cfgs, kpts=3000)
                Eval.eval_kitti_depth(net=net, datasets=datasets['test'], logger=log, cfgs=cfgs)
                print('END SAVE AND EVAL.')

    # save in the end
    log.save(net=net, state=state_dict, folder='model')
    Eval.eval_kitti_flow(net=net, datasets=datasets['test'], logger=log, cfgs=cfgs)
    Eval.eval_kitti_odometry(net=net, datasets=datasets['test'], logger=log, cfgs=cfgs, kpts=3000)
    Eval.eval_kitti_depth(net=net, datasets=datasets['test'], logger=log, cfgs=cfgs)

def run(cfgs):
    torch.backends.cudnn.benchmark = True
    # init logger
    log = logger.Logger(save_path=cfgs['log_path'], prefix=cfgs['log_name'], reuse=cfgs['reuse'], reuse_root=cfgs['reuse_root'])
    log.print_config(cfgs, write=True)

    # loading dataset
    print('Loading dataset...')
    if cfgs['dataset'] == 'kitti':
        datasets = load_kitti(cfgs)
    elif cfgs['dataset'] == 'kaist':
        datasets = load_kaist(cfgs)
    # model
    print('Loading model...')
    net = model.model_loader(cfgs)
    trainer(net, datasets, log, cfgs)
        
def main():
    parser = utils.prepare_parser()
    args = parser.parse_args()
    config = utils.ConfigFromFile(args.config)
    run(config)

if __name__ == '__main__':
    main()