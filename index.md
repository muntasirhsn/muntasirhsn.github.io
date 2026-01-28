# Selected projects in data science, machine learning, deep learning, and LLMs. 

---

## Neural Network-Based Time-Series Forecasting (CNN-LSTM)
This project implements a multi-step time-series forecasting model using a hybrid CNN-LSTM architecture. The 1D convolutional neural network (CNN) extracts spatial features (e.g., local fluctuations) from the input sequence, while the LSTM network captures long-term temporal dependencies. Unlike recursive single-step prediction, the model performs direct multi-step forecasting (Seq2Seq), outputting am entire future sequence of values at once. Trained on historical energy data, the model forecasts weekly energy consumption over a consecutive 10-week horizon, achieving a Mean Absolute Percentage Error (MAPE) of 10% (equivalent to an overall accuracy of 90%). The results demonstrate robust performance for long-range forecasting, highlighting the effectiveness of combining CNNs for feature extraction and LSTMs for sequential modeling in energy demand prediction.

<iframe src="images/forecasting_2.html"
        width="650"
        height="350"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Actual and predicted energy usage over 10 weeks of time period.


[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/TensorFlow-white?logo=TensorFlow)](#) [![](https://img.shields.io/badge/-Keras-white?logo=Keras&logoColor=black)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#)

[View sample codes on GitHub](https://github.com/muntasirhsn/CNN-LSTM-model-for-energy-usage-forecasting)



---

## MLOps with AWS: End-to-End ML Pipelines and Deploymen
Develop an end-to-end machine learning (ML) workflow with automation for all the steps including data preprocessing, training models at scale with distributed computing (GPUs/CPUs), model evaluation, deploying in production, model monitoring and drift detection with Amazon SageMaker Pipeline - a purpose-built CI/CD service.


<img src="images/MLOps6_Muntasir Hossain.jpg?raw=true"/> Figure: ML orchestration reference architecture with AWS

<img src="images/Sageaker Pipeline5.png?raw=true"/> Figure: CI/CD pipeline with Amazon Sagemaker 

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/)  [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) [![Amazon API Gateway](https://img.shields.io/badge/API_Gateway-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/api-gateway/) 

[View sample codes on GitHub](https://github.com/muntasirhsn/MLOps-with-AWS)



---


## Fine-tuning LLMs with ORPO & QLoRA (Mistral-v0.3)
ORPO (Odds Ratio Preference Optimization) is a single-stage fine-tuning method to align LLMs with human preferences efficiently while preserving general performance and avoiding multi-stage training. This method trains directly on human preference pairs (chosen, rejected) without a reward model or reinforcement learning (RL) loop, reducing training complexity and resource usage. However, fine-tuning an LLM (e.g. full fine-tuning) for a particular task can still be computationally intensive as it involves updating all the LLM model parameters. Parameter-efficient fine-tuning (PEFT) updates only a small subset of parameters, allowing LLM fine-tuning with limited resources. Here, I have fine-tuned the [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) foundation model with ORPO and QLoRA (a form of PEFT), by using NVIDIA L4 GPUs. In QLoRA, the pre-trained model weights are first quantized with 4-bit NormalFloat (NF4). The original model weights are frozen while trainable low-rank decomposition weight matrices are introduced and modified during the fine-tuning process, allowing for memory-efficient fine-tuning of the LLM without the need to retrain the entire model from scratch.  

[Check the model on Hugging Face hub!](https://huggingface.co/MuntasirHossain/Orpo-Mistral-7B-v0.3)


---
## Evaluating Safety and Vulnerabilities of LLM apps

### Overview
This project demonstrates iterative red-teaming of a policy assistant designed to answer questions about a government-style digital services policy, while strictly avoiding legal advice, speculation, or guidance on bypassing safeguards. The focus is on safety evaluation, failure analysis, and mitigation, rather than model fine-tuning.

### Model Separation Strategy
The system intentionally uses **different models for generation and evaluation**:
* Query responses are generated using **gpt-4o-mini**
* Safety evaluation is performed using **gpt-4o** via Giskard detectors
This reflects common red-teaming practice: lighter models are sufficient for generation, while **stronger models provide more reliable safety judgments**. Separating generation and evaluation also avoids self-evaluation effects and keeps evaluation costs controlled.

### Initial Evaluation
The assistant was evaluated using **Giskard** across prompt-injection, misuse, and bias detectors. The scan identified multiple failures where the agent did not attempt to answer questions based on the provided policy document. These were not hallucinations or unsafe outputs, but overly conservative refusals.

<img src="images/giskard1.png?raw=true"/> Figure 1: Initial scan results from Giskard.

### Analysis
The root cause was **over-refusal**.
The safety layer correctly blocked requests involving legal advice, speculation, or bypassing safeguards, but also refused some benign questions that could have been partially answered using neutral policy language. This reduced policy grounding and triggered Giskard failures.

### Mitigation
The refusal strategy was refined to better distinguish between:
* questions requiring refusal, and
* questions that can be answered safely using policy text alone.
Refusals were standardized using fixed, auditable messages, while benign queries now trigger policy-based responses where possible. Safety guarantees were preserved.

### Outcome
A follow-up Giskard scan showed improved behavior:
* fewer false positives for “did not attempt to answer”
* stronger grounding in policy text
* no regression in prompt-injection or misuse resistance

<img src="images/giskard2.png?raw=true"/> Figure 2: Post mitigation scan results from Giskard.

This project demonstrates a complete red-teaming loop — evaluation, failure analysis, mitigation, and re-evaluation — and shows how safety behavior can be systematically improved without increasing risk or cost.

[View project and source codes on GitHub](https://github.com/muntasirhsn/Red-Teaming-a-Policy-Assistant-with-Giskard)

---



## Analysis & Interactive Visualisation of Global CO₂ Emissions

The World Bank provides data for greenhouse gas emissions in million metric tons of CO₂ equivalent (Mt CO₂e) based on the AR5 global warming potential (GWP). It provides information on environmental impact at both national, regional and economic levels over the past six decades. 
### Analytical approach:
Time-series aggregation and normalisation across countries, regions, and income groups; comparative cohort analysis across geographic and economic categories; interactive filtering and visual exploration to support exploratory analysis and pattern discovery.

Some key insights from the data:  
* Many countries (e.g. China, India) and regions show a clear upward trend in carbon dioxide emissions from 1960 to 2024. For instance, although the population of China increased from 0.82 billion in 1970 to only 1.41 billion in 2023, emission drastically increased from 909 Mt CO₂e to over 13000 Mt CO₂e, reflecting rapid industrialization. While the population of China and India in 2023 were nearly the same, the CO₂ emission from China was 4.5 fold to that of India. 
* There is significant variation between countries. Highly industrialized or resource-rich nations (e.g., Saudi Arabia, United Arab Emirates) emit far more CO₂ than smaller or less industrialized countries (e.g., Aruba, Burundi).
* The data suggests a strong link between economic development and emissions growth. Countries experiencing rapid economic expansion (e.g., Vietnam, United Arab Emirates) show marked increases in emissions, while some developed countries (e.g., Germany, Austria, Belgium) have stabilized or slightly reduced their emissions in recent years, likely due to policy interventions or shifts to cleaner energy.
* While ‘High Income’ regions dominated the CO2 emissions pre-2020, the ‘Middle Income’ and ‘Upper Middle Income’ regions rapidly increased CO2 emissions after 2000, exceeding the emissions from ‘High Income’ regions.

### Global CO₂ emissions
<iframe src="images/co2_emissions_world_animation_start_on_2023_fixed.html"
        width="700"
        height="650"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Interactive visualization of global CO₂ emissions by country and year


### Time sereis CO₂ emissions
<iframe src="images/co2_emissions_timeseries_trend.html"
        width="650"
        height="550"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Time sereis CO₂ emissions for selected countries



### Population Growth
<iframe src="images/Population_timeseries.html"
        width="650"
        height="550"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Population growth for selected countries



### CO₂ emissions by income groups
<iframe src="images/co2_emissions_bar_income_zone_cleaned.html"
        width="650"
        height="550"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Interactive visualization of CO₂ emissions for different income zones from 1970 to 2023



### CO₂ emissions by geographic regions
<iframe src="images/co2_emissions_pie_fixed_colors_position.html"
        width="600"
        height="550"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Interactive visualization of CO₂ emissions for different geographic regions from 1970 to 2023



---

## Computer Vision: Building and Deploying YOLOv8 models for object detection at scale
Deployed a state-of-the-art YOLOv8 object detection model to real-time Amazon SageMaker endpoints, enabling scalable, low-latency inference for image and video inputs. Focused on model serving, endpoint configuration, and operational inference rather than model training.

<img src="images/highway1-detect3.gif?raw=true"/> Figure: Object detection with YOLOv8 model deployed to a real-time Amazon SageMaker endpoints.

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#) [![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-white)](https://github.com/AlexeyAB/darknet) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/) [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) 

[View project on GitHub](https://github.com/muntasirhsn/Deploying-YOLOv8-model-on-Amazon-SageMaker-endpoint)



---


<p style="font-size:11px">
