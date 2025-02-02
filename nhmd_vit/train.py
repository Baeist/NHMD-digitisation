import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from nhmd_vit.VitCTC import ViTCTC
import nhmd_vit.utils.device as devutils
from itertools import groupby
from nhmd_vit.data_processors.nhmd_datamodule import NHMDDataModule
from nhmd_vit.nhmdtokenizer import NHMDTokenizer
from transformers import RobertaTokenizer


# os.environ["CUDA_VISIBLE_DEVICES"]="0"
DATASET_PATH = "./data/NHMD_train_final"
CHECKPOINT_PATH = "./nhmd_vit/saved_models/"
device = devutils.default_device()
cp_file = 'run_name=0_epoch=2-val_cer=0.81.ckpt'
model = 'nhmddeit_small_patch16_384'
# model = 'nhmdbeit_base_patch16_384'
# model = 'nhmdbeit_large_patch16_384'
run_name = 'TROCR_deit_small_271k_trocrinit'
max_len = 50

def train_hybrid(**kwargs):
    pl.seed_everything(0)
    root_dir = os.path.join(CHECKPOINT_PATH, run_name)
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer = NHMDTokenizer()
    data_module = NHMDDataModule(
        data_path=DATASET_PATH,
        tokenizer=tokenizer, 
        max_len=max_len,
        train_bs=64,
        val_bs=64,
        test_bs=32,
        num_workers=16)
    data_module.setup()
    train_dl = data_module.train_dataloader()
    valid_dl = data_module.val_dataloader()
    os.makedirs(root_dir, exist_ok=True)

    wandb_logger = WandbLogger(project='NHMD', name=run_name)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min"),
                                    ModelCheckpoint(dirpath=root_dir,
                                                    filename='{run_name}_{epoch}-{val_cer:.2f}',
                                                    save_weights_only=False,
                                                    mode="min",
                                                    monitor="val_loss")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                        #  strategy='ddp',
                         precision='16-mixed',
                         max_epochs=30,
                         log_every_n_steps=10,
                         gradient_clip_val=5,
                         logger=wandb_logger,
                         fast_dev_run=False)


    trainer.logger._default_hp_metric = None
    pretrained_filename = os.path.join(root_dir, cp_file)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ViTCTC.load_from_checkpoint(pretrained_filename)
        trainer.fit(model, train_dl, valid_dl)
    else:
        print("Starting from scratch...")
        model = ViTCTC(max_iters=trainer.max_epochs*len(train_dl), tokenizer = tokenizer, **kwargs)
        trainer.fit(model, train_dl, valid_dl)

    # Test best model on validation and test set
    val_result = trainer.test(model, valid_dl, verbose=False)
    result = {"val_loss": val_result[0]["test_loss"], "val_cer": val_result[0]["test_cer"]}

    (x_test, y_test, _) = next(iter(valid_dl))
    pred = model(x_test, max_len)
    pred = pred.permute(1, 0, 2)
    logits = pred.log_softmax(2)
    _, max_index = torch.max(logits, dim=2)
    
    # Loop through the single batch
    for i in range(x_test.shape[0]):
        raw_prediction = list(max_index[:, i].detach().cpu().numpy())
        prediction_idxs = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != 0])
        try:
            # try to stop at eos_token_id for a specific line in batch
            idx = prediction_idxs.index(1)
        except:
            decoded_text = prediction_idxs
        else:
            decoded_text = prediction_idxs[: idx + 1]
        # Label text
        pred_str = tokenizer.decode(decoded_text, skip_special_tokens=True)
        label_str = tokenizer.decode(y_test[i], skip_special_tokens=True)
        print("Actual: " + str(label_str))
        print("Predicted: " + str(pred_str))
        break
    
    model = model.to(device)
    trainer.save_checkpoint(os.path.join(root_dir, f'{run_name}_final.ckpt'))

    return model, result

if __name__ == '__main__':
    model, result = train_hybrid(    model = model,
                                num_labels = 154, # alphabet size
                                        lr = 1e-4,
                                    warmup = 50,
                                   max_len = max_len)

    print(result)
    print(f"Validation CER:  {(100.0 * result['val_cer']):4.2f}%")

    
