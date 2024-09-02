import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import lr_scheduler
from preprocess import preprocess_v1
from utils import plot_utils
from model import model_v1
import torch
import torch.nn as nn
from model import trainer


if __name__ == "__main__":
    # Load Parameter about Radar & Preprocess
    HR_config = OmegaConf.load('config/radar/HR.yaml')
    preprocess_v1_config = OmegaConf.load('config/preprocess/v1.yaml')
    preprocess_module = preprocess_v1.preprocess_v1(HR_config.config, preprocess_v1_config.config)

    # Define HR, RR Model & Criterion & Optimizer
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    heart_model = model_v1.model_v1(device)
    resp_model = model_v1.model_v1(device)
    criterion = nn.MSELoss()

    optimizer_hr = torch.optim.AdamW(heart_model.parameters(), lr=0.001)
    scheduler_hr = lr_scheduler.CosineAnnealingLR(optimizer_hr, T_max=50)
    optimizer_rr = torch.optim.AdamW(resp_model.parameters(), lr=0.001)
    scheduler_rr = lr_scheduler.CosineAnnealingLR(optimizer_rr, T_max=50)

    # Define Sleep Stage Classification Model & Criterion & Optimizer
    sleep_model = model_v1.create_sleep_stage_classifier(device)
    sleep_criterion = nn.CrossEntropyLoss()
    sleep_optimizer = torch.optim.AdamW(sleep_model.parameters(), lr=0.001)
    sleep_scheduler = lr_scheduler.CosineAnnealingLR(sleep_optimizer, T_max=50)

    data_provider = preprocess_v1.data_provider(preprocess_module, device)

    # Define DataLoader about HeartRate & RespiratoryRate
    trainloader_hr, testloader_hr = data_provider.make_hr_dataloader()
    trainloader_rr, testloader_rr = data_provider.make_rr_dataloader()

    # Train HR Model & RR Model
    trainer.train_model(heart_model, device, trainloader_hr, criterion, optimizer_hr, 200, scheduler_hr)
    trainer.train_model(resp_model, device, trainloader_rr, criterion, optimizer_rr, 200, scheduler_rr)

    
    # Inference_HR & Plot the Comparison of Prediction
    inference_label_hr = trainer.evaluate_model(heart_model, device, testloader_hr, criterion)
    
    gt_label_hr = []
    for data in testloader_hr:
        gt_label_hr.append(data[1])
    plot_utils.plot_compare_output(inference_label_hr, gt_label_hr, 25, 200)

    
    # Inference_RR & Plot the Comparison of Prediction
    inference_label_rr = trainer.evaluate_model(resp_model, device, testloader_rr, criterion)
    
    gt_label_rr = []
    for data in testloader_rr:
        gt_label_rr.append(data[1])
    plot_utils.plot_compare_output(inference_label_rr, gt_label_rr, 0, 20)


    del trainloader_hr, testloader_hr, trainloader_rr, testloader_rr

    # Define HR, RR Extractor to Predict Sleep Stage
    heart_rate_model_extractor = model_v1.model_v1(device, extractor = 1)
    respiration_model_extractor = model_v1.model_v1(device, extractor = 1)
    
    heart_rate_model_extractor.load_state_dict(heart_model.state_dict())
    respiration_model_extractor.load_state_dict(resp_model.state_dict())

    # Define DataLoader about Sleep
    trainloader_sleep, testloader_sleep = data_provider.make_sleep_dataloader(heart_rate_model_extractor, respiration_model_extractor)

    torch.cuda.empty_cache()
    # Train Sleep Model
    trainer.train_model_clf(sleep_model, device, trainloader_sleep, sleep_criterion, sleep_optimizer, 200, sleep_scheduler)

    
    # Inference_sleep & Plot the Comparison of Prediction
    inference_label_sleep = trainer.evaluate_model_clf(sleep_model, device, testloader_sleep, sleep_criterion)
    
    gt_label_sleep = []
    for data in testloader_sleep:
        gt_label_sleep.append(data[1])
    plot_utils.plot_compare_output(inference_label_sleep, gt_label_sleep[:][0][0], 0, 6)


