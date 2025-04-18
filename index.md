## Selected demos in machine learning (ML), predictive modelling, deep learning, generative AI, NLP and computer vision. 

---

### MLOps with AWS: Train and deploy ML models at scale with automated pipelines
Develop an end-to-end machine learning (ML) workflow with automation for all the steps including data preprocessing, training models at scale with distributed computing (GPUs/CPUs), model evaluation, deploying in production, model monitoring and drift detection with Amazon SageMaker Pipeline - a purpose-built CI/CD service.


<img src="images/MLOps6_Muntasir Hossain.jpg?raw=true"/> Figure: ML orchestration reference architecture with AWS

<img src="images/Sageaker Pipeline5.png?raw=true"/> Figure: CI/CD pipeline with Amazon Sagemaker 

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/)  [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) [![Amazon API Gateway](https://img.shields.io/badge/API_Gateway-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/api-gateway/) 

[View codes on GitHub](https://github.com/muntasirhsn/MLOps-with-AWS)


---


### Energy usage forecasting with CNN-LSTM deep neural networks
Long short-term memory (LSTM) is a special type of recurrent neural network (RNN) that can be used for time-series forecasting. LSTM networks are capable of learning features from input sequences of data and can be used to predict multi-step sequences. In this example, I demonstrated a CNN-LSTM architecture for multistep time-series energy usage forecasting. A one-dimensional convolutional neural network (CNN) is used to read and encode the input sequence. An LSTM network is then used as a decoder to make a one-step prediction for each value in the output sequence. 

<img src="images/time-series3.png?raw=true"/>  Figure: Actual and predicted energy usage over 12 weeks of time period

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/TensorFlow-white?logo=TensorFlow)](#) [![](https://img.shields.io/badge/-Keras-white?logo=Keras&logoColor=black)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#)

[View codes on GitHub](https://github.com/muntasirhsn/CNN-LSTM-model-for-energy-usage-forecasting)


---


### Retrieval-Augmented Generation (RAG) with LLMs, embeddings, vector databases and LangChain
RAG is a technique that combines a retriever and a generative LLM to deliver accurate responses to queries. It involves retrieving relevant information from a large corpus and then generating contextually appropriate responses to queries. Here, I used the Llama 2 LLM and LangChain with GPU acceleration to perform generative question-answering (QA) with RAG.

![image](https://github.com/muntasirhsn/muntasirhsn.github.io/assets/29087240/cb5f2892-68a5-4d68-b9b4-568959b2595a) Figure: A schematic representation of RAG with a retriever and an LLM

To get a hands-on introduction to RAG, [view my codes on GitHub](https://github.com/muntasirhsn/Retrieval-Augmented-Generation-with-Llama-2)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/HuggingFace_Transformers-white?logo=huggingface)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#) 

**Try the app below that uses the Llama 3 8B model and FAISS vector store for RAG on your PDF documents!**

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.31.0/gradio.js"
></script>

<gradio-app src="https://muntasirhossain-rag-pdf-chatbot.hf.space"></gradio-app>


---


### Deployment and API integration of LLMs and ML models with FastAPI and AWS Lambda serverless services
FastAPI is a modern web framework designed for building APIs with Python.  It emphasizes simplicity and ease of use while providing robust functionality, making it popular among developers for creating high-performance APIs. 
In this example, I deployed a  Microsoft/Phi-2 LLM from the Hugging Face hub to an Amazon Sagemaker endpoint with GPU instances. To integrate the Sagemaker endpoint to consumer-facing APIs, I utilise FastAPI and AWS lambda as a serverless service.  

![image](https://github.com/muntasirhsn/muntasirhsn.github.io/assets/29087240/92bae7ad-6a12-4ff7-a026-2a17afcf7090) Figure: Deploying Amazon Sagemaker ML model endpoint with FastAPI and AWS Lambda

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/)  [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) [![Fast API](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](#)



---

### Computer Vision: Deploying YOLOv8 model at scale with Amazon Sagemaker Endpoints
YOLO (you only look once) is a state-of-the-art, real-time object detection and image segmentation model used in computer vision. The latest model YOLOv8 is  known for its runtime efficiency as well as detection accuracy. To fully utilise its potential, deploying the model at scale is crucial. Here, a YOLOv8 model was hosted on the Amazon SageMaker endpoint and inference was run for input images/videos for object detection.

<img src="images/highway1-detect3.gif?raw=true"/> Figure: Object detection with YOLOv8 model deployed to a real-time Amazon SageMaker endpoint

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#) [![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-white)](https://github.com/AlexeyAB/darknet) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/) [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) 

[View project on GitHub](https://github.com/muntasirhsn/Deploying-YOLOv8-model-on-Amazon-SageMaker-endpoint)


---


### Parameter-efficient fine-tuning of LLMs (Llama-3) with Quantized Low-Rank Adaptation (QLoRA)
Fine-tuning an LLM (e.g. full fine-tuning) for a particular task can be computationally intensive as it involves updating all the LLM model parameters. Parameter-efficient fine-tuning (PEFT) updates only a small subset of parameters, allowing LLM fine-tuning with limited resources. Here, I have fine-tuned the [Llama-3-8B](meta-llama/Meta-Llama-3-8B) foundation model with QLoRA, a form of PEFT, by using NVIDIA L4 GPUs. In QLoRA, the pre-trained model weights are first quantized with 4-bit NormalFloat (NF4). The original model weights are frozen while trainable low-rank decomposition weight matrices are introduced and modified during the fine-tuning process, allowing for memory-efficient fine-tuning of the LLM without the need to retrain the entire model from scratch.  

[Check the model on Hugging Face hub!](https://huggingface.co/MuntasirHossain/Meta-Llama-3-8B-OpenOrca)

Try the following conversational AI agent that uses the fine-tuned Llama 3 8B model in the backend! **Please note: the app is running on a basic CPU! A low-precision version of the fine-tuned LLM is deployed to overcome the hardware restrictions. Hence, performances may not be up to the model's full potential and responses may be slow!**

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.31.3/gradio.js"
></script>

<gradio-app src="https://muntasirhossain-fine-tuned-llama-3-8b-chatbot.hf.space"></gradio-app>


[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#) [![](https://img.shields.io/badge/HuggingFace_Transformers-white?logo=huggingface)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#) 


---


### Reinforcement Learning with Human Feedback (RLHF) 
Reinforcement Learning with Human Feedback (RLHF) is a cutting-edge approach used to fine-tune Large Language Models (LLMs) to generate outputs that align more closely with human preferences and expectations. Here, I utilised RLHF to further fine-tune a [Google Flan-T5 Large](https://huggingface.co/MuntasirHossain/flan-t5-large-samsum-qlora) and generate less toxic content. The model was previously fine-tuned for generative summarisation with PEFT. A [binary classifier model](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) from Meta AI was used as the reward model to score and reward the LLM output based on the toxicity level. The LLM was fine-tuned with Proximal Policy Optimization (PPO) using those reward values. The iterative process for maximising cumulative rewards and fine-tuning with PPO enables detoxified LLM outputs. To ensure that the model does not deviate from generating content that is too far from the original LLM, KL-divergence was employed during the iterative training process.

[Check the model on Hugging Face hub!](https://huggingface.co/MuntasirHossain/flan-t5-large-samsum-qlora-ppo)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#) [![](https://img.shields.io/badge/HuggingFace_Transformers-white?logo=huggingface)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#) 


---


### Data Engineering: Extract, transform, and load (ETL) in Python
This is a simple demonstration of an ETL process in python. Tesla share price data is extracted, transformed and loaded into a PostgreSQL database. An automation tool like Windows Task Scheduler or Airflow can be easily employed to execute the ETL process on a schedule and new records will be automatically extracted, transformed and loaded into the database!

<img src="images/PostgreSQL2.png?raw=true"/> Figure: Transformed and loaded data in PostgreSQL database

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/PostgreSQL-white?logo=PostgreSQL)](#) [![](https://img.shields.io/badge/pandas-white?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAAEDCAMAAABQ/CumAAAAeFBMVEX///8TB1QAAEb/ygDnBIgPAFLNzNYTAFnQ0NgMAFcAAETb2eP39/oUBlfV1N7/xwDmAID/9tfLydcjG17/4Yz//vbCwM3ykcL61OfoBIwyKmgAADYAAE0AAErx8PTIxdT/+un/34T85/Lyir/lAHv50eX+9fkpH2Ma8J+4AAACEklEQVR4nO3dzVIaQRSAUYNCEIGoiYmJivnP+79hFrmLVHELZ6pnmG483xqaPruh5lb32ZkkSZIkSZIkvb52z7dZU2+rT4uH2X6rx6m31afF7M1+87dTb6tPCDWEUEMINYRQQ5MS1tu0nqtMSrhKn26e1v1WmZawyn58g4DQL4QIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyECOFA6cvM5a4nYb29yjoO4WmVvM58WPQkbF8e+RqPcDlPVp4t+xLS/W0QEBCqI8yTLpsizN8n/WmJ0CEEBAQEBAQEBIT2CF+/fci6a4hw8y7rvC3CeRYCAgICAgICAgICAgICwlCEtJYIdzdp/3+kdkKHToFQ+RjJMCEcCKF7CAdC6B7CgRC6Nylh9zGtJUJ6uNCsnsOFhhkvPAHC9x+fsloi/Pp5nXTREuH++iLpMwICAgICAgICAgICAgKC/87R7/u0lggdQkBAQEBAQEB4dYQON67UTqh9KuwkDlRBQED4R8gOF5o3Rdh8yepLGO0ez6MNPO+WQ9w3NilhvBAihLIQIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyEKJt+lL0SNeADUR4TG9cGWXHew10AkPP4aRBO9ohEuOFUEMINYRQQwg1dAKEDvd41t5t2u7lL0qSJEmSJEnSyfUXeomSFq0EzbkAAAAASUVORK5CYII=)](#) [![](https://img.shields.io/badge/NumPy-white?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAmVBMVEX///9Nq89Nd89Iqc4+bsxFcs7R2vGVyuA7pcx9vtmWrODW3/Pg5vVCp82l0eQ7bMx2u9igz+Pq9Pn0+vzP5vCEwtyNxt5ktNTV6fJasNK12ene7vXD4O1uuNba7PTn8/ijtuS83eu2xelrjNZ9mdrt8fp1k9iIod1+mtpcgtJQetC/zOyywug0aMsjn8mbsOLH0u7m6/dvj9e9++DaAAAJe0lEQVR4nO3da1ujOhAAYEqKxlqkivfLWrfedffo/v8fd6BaW2AyMwkJoTyZb2e3cHh3aMhMgEZRiBAhQoQIESJEiBAhQoQIESJECCAuj3wfgds4EKk8HbCx8I1Go+EaC58YLUMM0viVv1UML49V3+CM+UFa9w3LCPsGY8wPpMo3CGN+qMzfjzE9zX0fpnHkh2j+tj6PjPxtdR4Ln2D6vvO4XUZN39adq/mutu/LuLstedxlDTCQ8e+172NnRr4rTZIoZ1e+j5wfBieqPPvl+6j1Qi+PIr049n3E+qFhTEfb8gWsRc4bc1Jx6ftIzYNhFOmB76MkI8dOsfwUnbylKXYR7Mf1sUiTvMCMR6fK76OQJ8hE5vysD3OA7+FEokOhyij3btUbXc1k+U/g2bgxXMoz1HjSNKIXwJ8NBHoaO47a5YAwVvMo0E9XPizkoR8jMG3BjbfrPAr0AljsuTo4ecmj4nIuz86RjVbGdIRdAA+ACV/neUSmnfKGMuIXwMtGZ3WVxw6NxHSMMO5hZ9zlSD1j78xIlw1C3piVCXeIrzMjrywScqZvvL6gd+3cyC8XijzqlbPHZ5K5Y4dGvbJW6JTsv254vjJSV3nUb02wjbczvu9rxw6MZv0zIfdoIzSjo8J6Hg1bSxwjUnngRpt5NOx/rogz9EiKEslw5/byePfXsP054n0VzzVGmdrO/1pqXxWVmuE/M/PSb2YUaHnZiRGfhleNzKuhI5+RES8A2xqFPLHr0zaKFG3dgMGd1TAvQibBNgp5cWfyP+AZneRvFb9YRrzDe4kdHm10lr9V0Ebcd0cthV6jRuvjCxS4ES/gvwokynihMjrP3yrURnxKfHyx2s7IaN03+U/9d/CYgy9TVy8HxK0JzWrYwfk5yZK5+m+Lgq5xCFherhoFErFsXzVS+ftlsno1SeIkmasPonqu4isQt3v6Od8wUr6rPblLcYAohHFhfFd/YiOP8gY5hdQFINHqvRuVG1Kz98JX7IfQQLEUxvE4+a3+zHce0Qk2XlsSSy/lxYXI3/IQ2ggLY4wYi7IAXXei7x0i2meXaHWyGvLaCUvjm/pzx9gEFOrQA3k0LGXXQ3pbYWF8+TDYhapD34g0PdA3bl6y2gvjONM2Yh16yKi39+pQbkNYGnc0Nqc69O2M9WmVHWFhXHDzyOnQN42CaWxOG20JC+Mj5/tyrJw74yFYxuZ0yqYwThacIzDyfR1qeqjvsyqMM+LLeKTdU6qEnKHdq3PQZ1eYPBLbqQ6CEQJfXEW6jjaFcTaltjRt8VI+5OywKkzu6W1NjITvCj37rQrj8Sdja10j4YuiXXR6ZFeYPLG212nxMrrjh+iX264wzh54e+C2eFnd/06FyYS7D46RPD89COOEXwhQRvbqRsfCucZ+MKO8Ya/edCuMx1p7UhmJ/B1XjrpjIdafAg8WMFK+M3my+d8dC+NX3b3Vyw3aNxJehWOkcaOIzVY9wzfyLNRPYrQ2snzehWOTzlRppFa/119Zz8L4RbXdzmIf2es1fn3YHJJ8C5WV8H6WPZIFFhzVIde3MPmn2G5/HCdGxvqqoW+hshIuhLGBsbkq6l2YPCNCXSO06utdGGdwJfwt1DEqVrW9CxWV8I+wNN4zjKo7E/wL4zFYCW8IOcbrC1V7Qux5F8KVcEVIGdV3lvRCGGdQJVwTYkZ1/voiBCvhhrA0PgOjErV60wch2M4AhMUH/zzX8kivvvVCOAYqYVBYGivTVfD2E2OhkCZPS3OEccwWxuPKRHaPXttgC7WeXdEVApVw10Lj58iYOWwWUR0Lme1Wc2GziOpU2MLHFcaNNeEOhZo3yxsKs3pF35mwpY8tbKwJdyTE3+NgU9iohLsQCgs+vrC+JuxeaPIwRxthvRJ2LbTl0xDWiii3QpGOjB5WaSWsrQm7FAr8bQzOhEnlrn53Qrs+HWG1iHIlTK2/TgoWvoLCeRdCrEAyfRoBsLyDx5+4F54j9w60eRqhHtnnI/TH44173R0JUV+rpxFqh/k5zaDD31hO7Fj49bCKReE0ugf/fF0JdypcPYxjVUglsUPh+mEjq8LoGfyC/qwJdybceO+UZeEnmMSfStiWUEhUWH2Yyq4wegKTuKqE7QiFROef9bdpWBY+QEn8ucXdhrCoH3R81oWKS+XUlpCoH6C3odgWPoCN+3s7Qjp/wHKObaFqvmNBSOQvB30OhDn4TXxqLSTqd/X7Xq0LozmYxIeWQsqnXq2yL8zBwmPSSoj3z/CHbe0L4SSOc3Mh1R88RO+mdiCMwCTOTYV0//OgoxXStfA3eMXIjYTyjO4Pdi+MoH5GuSasLcTfQexR+AZKtIXc9RUPwugF+vs3PSF//ciH8AOivOgIifcOV5prPoRgErOPfbB8BITk+7Fn3oU7kGUxZeVwRr8bW/gXRgvgEwk4F2gIT+n3m/dBCCYR7Io3hJjvu/7rgzBa8Nc2uMJ1fdsLoWpUMRZu1re9EEZgj99YWK3f+yGE28Nmwnp/oh9CuMdvImz2X3oi/FRc/TSFR8BvtfRECLeHNYVw/6wvQrA9rCWEff0Rstf7FUKVr0dCsD3MFap9PRJykwgIMV8Hwo8EOHJImPOS2BBSb2N3Loyi+bhhhIRwZ5ES4vnrSBjlk7oRFILtYUJ4yHgaQUMoDIXFMPKUVQ4fFPKSWKuAGavcfGEqW9zON73fNMLCSD+HnHV8rrD1TwjsL9YXdYUQvlPKrlD58+U2fiJh52VlVAg5SXQjtPYzF2+vY1QI9vjdC43emamK9+WwqhKCPX7XwlRa9JUxL6YASuEHOQG3LdR+HygjHibZH+UTr1B72KHQ6vm5abxXvvMC7Cy6ErrIHx3/iMu+PaGr/FFBdRZtCf3kbxlEEu0I2e+pdRFEErWFo5vNDZZCj/lbBt5ZbC/0mr9lwDee2hJKT+NLJdDOYluh7/wtA01iS2FPAkviMIRYe3gYQqyzOBAheLvbsITqKkp/TtNPoXqw0RVaf9zQWtjJoe/5GRaqNVMdof/5GRrT5mKAnrDP+fuK/B4abrjCnufvOz7iZhp5wu3wlTFppJEh5P1GSV9i/2WsK9wqXxnv1RGHEIr+jy/NqI44uHDr8vcdO68JR1jkr6/zFzrWK+VKoe3XXXQdn48ZKhT9nX+y4/viCAq3PX+rmJQL5YBwKL4ipousKRyQr4y35E9NOCxfEflT5U38N9s/vhDR8nV5IUKECBEiRIgQIUKECBEiRIgQg4v/AeWW3tnJVfCfAAAAAElFTkSuQmCC)](#)

[View code on GitHub](https://github.com/muntasirhsn/ETL-in-Python)

---






---
<p style="font-size:11px">
