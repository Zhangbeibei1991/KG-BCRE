import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransH, RotatE
from openke.module.loss import MarginLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default="biocause", help="task name")

args = parser.parse_args()

KG_PATH = f'./{args.task_name}-triplets/'
# dataloader for training
train_dataloader = TrainDataLoader(
    in_path=KG_PATH,
    nbatches=128,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0)

# dataloader for test
test_dataloader = TestDataLoader(KG_PATH, "link", type_constrain=False)


def get_model():
    '''
    Return the KGE model given model type and dimension.
    args: model_type: str
    args: loss_type: str
    args: dim: int

    returns: model: one of the KGE model.
    '''
    assert model_type in ['transe', 'transh', 'rotate'], "Unknown model type!!!"

    # obtain loss function
    loss = get_loss(loss_type)

    if model_type == 'transe':
        model = TransE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=dim,
            p_norm=1,
            norm_flag=True)

    elif model_type == 'transh':

        model = TransH(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=dim,
            p_norm=1,
            norm_flag=True)

    elif model_type == 'rotate':

        model = RotatE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=dim,
            margin=6.0,
            epsilon=2.0)

    else:
        raise NotImplementedError

    return model


def get_ns(model):
    if loss_type == 'margin':
        ns = NegativeSampling(
            model=model,
            loss=MarginLoss(margin=5.0),
            batch_size=train_dataloader.get_batch_size(),

        )
    elif loss_type == 'sigmoid':
        ns = NegativeSampling(
            model=model,
            loss=SigmoidLoss(adv_temperature=1),
            batch_size=train_dataloader.get_batch_size(),
            regul_rate=0.0
        )
    else:
        raise NotImplementedError
    return ns


def get_loss(loss_type):
    '''
    Return the loss function given the loss type.
    args: loss_type: str

    returns: loss: one of the KGE model.
    '''

    assert loss_type in ['margin', 'sigmoid'], "Unknown model type!!!"

    if loss_type == 'margin':
        loss = MarginLoss(margin=args.margin)

    elif loss_type == 'sigmoid':
        loss = SigmoidLoss(adv_temperature=1)

    return loss


def train():
    '''
    Model trainning
    '''
    model = get_model()
    ns = get_ns(model)
    train_times = 500

    best_mrr = 0
    improved = True

    iterations = 1
    # Of the checkpoint improve from the last one, keep training. Otherwise, terminates.
    while improved:
        improved = False
        # train the model
        trainer = Trainer(model=ns, data_loader=train_dataloader, train_times=train_times, alpha=1.0, use_gpu=True,
                          opt_method=opt_method)
        trainer.run()
        tester = Tester(model=model, data_loader=test_dataloader, use_gpu=True)
        mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)
        if mrr > best_mrr:
            print(f"MRR improved from {best_mrr} to {mrr}")
            improved = True
            best_mrr = mrr
            model.save_checkpoint(
                f'kge_embed/UMLS_CUI_STY_tmp_{args.task_name}_{model_type}_{loss_type}_d{dim}_{opt_method}-{alpha}_t{train_times * iterations}.ckpt')
        iterations += 1


if __name__ == "__main__":
    os.makedirs('kge_embed', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default='biocause')
    parser.add_argument("--loss_type", type=str, default='margin')
    parser.add_argument("--model_type", type=str, default='transe')
    parser.add_argument("--alpha", type=float, default=0.5)
    # parser.add_argument("--dim", type=int, default=300)
    parser.add_argument("--dim", type=int, default=300)
    parser.add_argument("--opt_method", type=str, default='adam')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()

    # parse args
    loss_type = args.loss_type
    model_type = args.model_type
    alpha = args.alpha
    dim = args.dim
    opt_method = args.opt_method
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train()
