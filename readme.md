# The official implementation of STD-PLM.

### Train and Test the STD-PLM

```bash
cd ./code/STD-PLM/src

#imputation
bash '../scripts/pems08_sctc30.sh'


### Arguments

```
--lora : Fine-tune the PLM with lora
--ln_grad : Train the layernorm of PLM
--wo_conloss  : Removing the constraint loss function
--sandglassAttn : Introducing the SGA module.
--time_token : Add time token
--model plm : Specify the PLM to use
--llm_layers layers : Specify the layers of PLM
--few_shot ratio    :   Specify the ratio of few-shot
--zero_shot :   zero-shot
--from_pretrained_model model.pth   : Specify the trained model weights
--task task : Specify the task
```
