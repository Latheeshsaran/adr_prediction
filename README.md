<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <hgroup>
  <h1 align="center">ADR-FORECAST <br> PREDICTION OF ADVERSE DRUG REACTIONS</h1>
  <p>By Team Hawkeye</p>
  </hgroup>

  <p align="center">
    Welcome to our project!
    <br />
    <br />
    <a href="https://colab.research.google.com/github/lovelindhoni/adr_prediction/blob/main/adr_prediction.ipynb">View in Colab</a>
    ·
    <a href="https://adr-forecast.vercel.app/">View Deployment</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
      <ul>
        <li><a href="#inspiration">Inspiration</a></li>
        <li><a href=#social-impact>Social Impact</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#intel-oneapi">Intel® OneAPI</a>
      <ul>
        <li><a href="#intel-oneapi">Use of oneAPI in our project</a></li>
      </ul>
    </li>
    <li><a href="#what-it-does">What it does</a></li>
    <li><a href="#how-we-built-it">How we built it</a></li>
    <li><a href="#what-we-learned">What we learned</a></li>
    <li><a href="#references">References/a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<div align="center">
  <img src="https://ik.imagekit.io/lovelin/bfb70652-e79e-455e-86a2-fc8802302e8e.jpeg?updatedAt=1710040380234" type="gif" alt="png" width="750">
</div>
<br>
<br>
<br>
An adverse drug reaction (ADR) is when a medication causes unexpected and unwanted effects in your body.The ADRForecast addresses the issue of adverse drug reactions (ADRs) in polypharmacy by predicting potential side effects. It uses a deep learning framework based on drug-induced gene expression signatures to enhance patient safety and drug therapy efficacy. The model is effective in predicting ADRs for unseen drug interactions, outperforming other methods. Additionally, it can predict ADRs for new compounds not used in training, contributing to improved accuracy and understanding of drug-induced gene expression signatures and drug interaction mechanisms.                                                                                                                                                                                                                                                               

#### Intel® oneAPI is used to optimize the models to provide accurate and efficient prediction,

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Inspiration 

The inspiration behind the creation of Adverse Drug Reaction is
Unlocking Insights, Saving Lives: Predicting Adverse Drug Reactions with Precision.
Empowering Healthcare with Data: A Journey Towards Safer Medications.
Shaping the Future of Medicine: Harnessing AI to Anticipate Adverse Drug Reactions.
From Data to Decisions: Illuminating Adverse Drug Reactions for Better Patient Care.
Transforming Healthcare: Predicting Adverse Drug Reactions for Enhanced Patient Safety.
Driving Innovation, Protecting Patients: Predicting Adverse Drug Reactions with AI.

### Social Impact 
The social impact of predicting adverse drug reactions (ADRs) is profound and far-reaching. By harnessing advanced technologies to anticipate potential adverse effects of medications, we can significantly improve patient safety and healthcare outcomes. This predictive capability empowers healthcare providers to make more informed decisions when prescribing medications, reducing the risk of harmful reactions and mitigating potential health complications for patients. This not only enhances individual patient care but also contributes to broader public health initiatives by minimizing the incidence of adverse drug events on a population level. Ultimately, the ability to predict ADRs has the potential to save lives, alleviate suffering, and enhance the overall quality of healthcare delivery.

### Built With 
The power of Intel oneAPI, Python, Jupyter, TensorFlow, flask, sveltekit and Gradio to create an innovative solution for predicting brain diseases using machine learning. Python's versatility and readability serve as the foundation, while Jupyter notebooks facilitate interactive model development. TensorFlow powers the machine learning model construction and training. Intel oneAPI enhances computational performance, enabling efficient predictions. Gradio and sveltekit simplifies the deployment process by transforming the project into an interactive web application, allowing users to input data and receive predictions seamlessly.

* [![oneapi][oneapi]][oneapi-url]
* [![python][python]][python-url]
* [![jupyter][jupyter]][jupyter-url]
* [![tensorflow][tensorflow]][tensorflow-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Intel one api -->
## Intel® oneAPI
Intel® OneAPI is a comprehensive development platform for building high-performance, cross-architecture applications. It provides a unified programming model, tools, and libraries that allow developers to optimize their applications for Intel® CPUs, GPUs, FPGAs, and other hardware. Intel® OneAPI includes support for popular programming languages like C++, Python, and Fortran, as well as frameworks for deep learning, high-performance computing, and data analytics. With Intel® OneAPI, developers can build applications that can run on a variety of hardware platforms, and take advantage of the performance benefits of Intel® architectures.
<!-- Use of oneAPI in our project -->
### Use of oneAPI in our project

In this section, we'll outline how we utilized various Intel® oneAPI libraries and frameworks to enhance the performance and efficiency of our models.

* <b>Intel® oneAPI Data Analytics Library (oneDAL)</b>

The oneAPI Data Analytics Library (oneDAL) is a versatile machine learning library that accelerates big data analysis at all stages of the pipeline. To leverage the power of oneDAL, We employed the Intel® Extension for Scikit-learn*, an integral part of oneDAL that enhances existing scikit-learn code by patching it.

Installation:
<code>pip install scikit-learn-intelex</code> 

Usage:<br>
<code>from sklearnex import patch_sklearn
patch_sklearn()</code>

By integrating Intel® Extension for Scikit-learn*, We achieved substantial acceleration, with performance gains ranging from 10x to 100x across various applications.

* <b>Intel® oneAPI Deep Neural Network Library (oneDNN)</b>

To optimize deep learning applications on Intel® CPUs and GPUs, We integrated the oneAPI Deep Neural Network Library (oneDNN). To enable oneDNN optimizations for TensorFlow* running on Intel® hardware, We used the following code:

<code>os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['DNNL_ENGINE_LIMIT_CPU_CAPABILITIES'] = '0'</code> 

* <b>Intel® oneAPI DPC++ Library (oneDPL)</b>

The Intel® oneAPI DPC++ Library (oneDPL) aims to simplify SYCL* programming efforts across devices for high-performance parallel applications. We harnessed the power of oneDPL using specific environment variables to optimize performance and memory utilization.

<code>os.environ['ONEAPI_DEVICE_SELECTOR'] = 'opencl:*'
os.environ['SYCL_ENABLE_DEFAULT_CONTEXTS'] = '1'
os.environ['SYCL_ENABLE_FUSION_CACHING'] = '1'</code>

* <b>Intel® oneAPI AI Analytics Toolkit (AI Kit)</b>

The Intel® oneAPI AI Analytics Toolkit (AI Kit) offers an integrated solution for preprocessing, machine learning, and model development. To optimize deep learning training on Intel® XPUs and streamline inference, We utilized the toolkit's Intel®-optimized deep-learning frameworks for TensorFlow*.

<code>pip install --upgrade intel-extension-for-tensorflow[cpu]</code>

We set the backend type to CPU for Intel® Tensorflow Operator Optimization:

<code>os.environ['ITEX_XPU_BACKEND'] = 'CPU'</code>

And enabled Advanced Automatic Mixed Precision for improved inference speed and reduced memory consumption:

<code>os.environ['ITEX_AUTO_MIXED_PRECISION'] = '1'</code>

#### Model Specifics and Usage
ADR prediction model  is TensorFlow-based. For these, We used the Intel® Extension for TensorFlow* from the AI Kit, oneDAL, oneDPL and oneDNN to enhance performance..

### Performance Comparison
The following graphs illustrate the substantial performance improvements achieved by integrating Intel® oneAPI libraries and frameworks into our models:
1. Comparing execution time of model training for adr_prediction model <br><br>
<a href="https://github.com/lovelindhoni/adr_prediction.git">
    <img src="https://ik.imagekit.io/lovelin/Figure_1.png?updatedAt=1710048583447" >
</a><br><br>

By leveraging the power of Intel® oneAPI libraries and frameworks, our models achieves remarkable performance enhancements and optimized memory utilization across various disease prediction models. The seamless integration of oneDAL, oneDNN, oneDPL, and AI Kit contributes to faster training, efficient inference, and improved overall user experience.

<!-- What it does -->
## What it does 
The ADRForecast addresses the issue of adverse drug reactions (ADRs) in polypharmacy by predicting potential side effects. It uses a deep learning framework based on drug-induced gene expression signatures to enhance patient safety and drug therapy efficacy. The model is effective in predicting ADRs for unseen drug interactions, outperforming other methods. Additionally, it can predict ADRs for new compounds not used in training, contributing to improved accuracy and understanding of drug-induced gene expression signatures and drug interaction mechanisms.  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## How we built it 
These are the steps involved in making this project: 
* Importing Libraries
* Data Importing
* Data Exploration
* Data Cleaning
* Preparing the Data
  * Creating a Generator for Training Set
* Writing the labels into a text file 'Labels.txt'
* Model Creation
* Model Compilation
* Training the Model 
* Testing Predictions
* Saving model as 'adr_prediction.h5'
* Deploying the Model on huggingface and with a cool user-friendly frontend written in sveltekit.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## What we learned 
✅Building ADR - FORECAST using oneDNN and Intel oneAPIs has been a transformative journey, providing us with a deep understanding of cutting-edge technologies and their practical applications in the field of prediciting adverse drug reactions. Here's a summary of key learnings from this experience:

✅ Hardware Optimization Expertise: Working with oneDNN and Intel oneAPIs exposed us to advanced techniques for optimizing machine learning models on diverse hardware architectures. We gained insights into harnessing the full potential of CPUs, GPUs, and other accelerators, enhancing our ability to design efficient and high-performance solutions.

✅Hardware-Agnostic Deployment: The ability to deploy our models seamlessly on various hardware architectures showcased the power of hardware-agnostic solutions. We gained confidence in creating versatile applications that can adapt to different computing environments.

✅Model Evaluation:  Working with oneDNN and Intel oneAPIs encouraged us to iterate on model architectures and hyperparameters. We gained proficiency in fine-tuning models for optimal accuracy and performance, resulting in refined adverse drug reaction prediction capabilities.

✅Educational Impact: The project's use of advanced technologies like oneDNN and Intel oneAPIs presented opportunities for educational outreach. We learned to convey complex technical concepts to wider audiences, promoting awareness of AI's potential in healthcare.

✅Innovation at the Intersection: ADR - Forecast's creation at the intersection of medicine and technology highlighted the potential for innovative solutions that bridge disciplines. We gained insights into the challenges and rewards of interdisciplinary projects.


In conclusion, our journey of building ADR - Forecast using oneDNN and Intel oneAPIs has been a transformative experience that has enriched our understanding of cutting-edge technologies, healthcare applications, and the profound impact of responsible AI integration. This project has yielded a diverse array of insights, fostering growth in technical expertise, ethical considerations, collaboration, and real-world problem-solving. Through this endeavor, we have not only created a adverse drug reaction prediction platform but also embarked on a significant learning journey with enduring implications.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### References
<hr style="border: 0.5px solid #ddd;">

DrugBank - https://go.drugbank.com/

Gene Expression Omnibus - https://www.ncbi.nlm.nih.gov/geo/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->



[python]: https://img.shields.io/badge/Python-3470a3?&logoColor=white
[python-url]: https://www.python.org/
[jupyter]: https://img.shields.io/badge/Jupyter%20Notebook-da5b0b?&logoColor=white
[jupyter-url]: https://jupyter.org/
[tensorflow]: https://img.shields.io/badge/TensorFlow-f0b93a?&logoColor=white
[tensorflow-url]: https://www.tensorflow.org/
[gradio-url]: https://www.gradio.app/
[oneapi]: https://img.shields.io/badge/Intel%20oneAPI-20232A?&logoColor=61DAFB
[oneapi-url]: https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-0/intel-oneapi-data-analytics-library-onedal.html
[onednn]: https://img.shields.io/badge/oneDNN-20232A?&logoColor=61DAFB
[onednn-url]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html
