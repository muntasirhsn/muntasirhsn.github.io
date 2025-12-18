# Selected projects in data science, machine learning, deep learning, and computer vision. 

---

## Exploratory Spatial and Environmental Data Analysis with Interactive Visualisation

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

## Advanced forecasting models with deep CNN-LSTM neural networks
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

## Train and deploy ML models at scale with automated pipelines (MLOps with AWS)
Develop an end-to-end machine learning (ML) workflow with automation for all the steps including data preprocessing, training models at scale with distributed computing (GPUs/CPUs), model evaluation, deploying in production, model monitoring and drift detection with Amazon SageMaker Pipeline - a purpose-built CI/CD service.


<img src="images/MLOps6_Muntasir Hossain.jpg?raw=true"/> Figure: ML orchestration reference architecture with AWS

<img src="images/Sageaker Pipeline5.png?raw=true"/> Figure: CI/CD pipeline with Amazon Sagemaker 

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/)  [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) [![Amazon API Gateway](https://img.shields.io/badge/API_Gateway-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/api-gateway/) 

[View sample codes on GitHub](https://github.com/muntasirhsn/MLOps-with-AWS)



---








---


<p style="font-size:11px">
