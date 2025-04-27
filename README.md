# Credit Card Fraud Detection
## Overview
This project focuses on detecting fraudulent credit card transactions using a range of Machine Learning and Deep Learning models. Models used include Random Forest, XGBoost, LSTM, Autoencoders, Isolation Forest, and DBSCAN.
The aim is to build robust classifiers, evaluate them using ROC-AUC scores, and compare their performances to identify the most effective fraud detection techniques.

#### Technologies Used
1. Python

2. Scikit-learn

3. TensorFlow / Keras

4. XGBoost

5. Matplotlib & Seaborn (for visualization)

#### Key Steps
1. Data Preprocessing

2. Train-Test Split

3. Scaling and Standardization

4. Model Building:

5. Random Forest Classifier

6. XGBoost Classifier

7. LSTM Neural Network

8. Autoencoders

9. Isolation Forest


#### Model Evaluation:

1. Confusion Matrix

2. Precision, Recall, F1-Score

3. ROC-AUC Curve

4. Performance Comparison

### Observations:
1. Random Forest, LSTM, XGBoost provide better result when considering the ROC-AUC score as the evaluation metric for credit card fraud detection.
2. Isolation Forest and Autoencoder perform better accuracy when the number of anomalies is significantly lower compared to valid transactions.
3. However, since the dataset used in this study is fairly balanced, the accuracy of Isolation Forest and Autoencoders is relatively low.
4. Comparing the ROC-AUC scores of different models, LSTM is 99.26% accurate, outperforming Random Forest which is 98.54% and XGBoost at 98.89% .
5. The accuracy can be further improved by increasing the sample size or using more advanced deep learning algorithms, though this comes at a higher computational cost. Additionally, implementing complex anomaly detection models could enhance fraud detection accuracy.

### Future Enhancements
1. Hyperparameter Tuning:
Apply advanced techniques like Grid Search and Bayesian Optimization to fine-tune model parameters for even better performance.

2. Handling Imbalanced Data:
Experiment with Synthetic Minority Over-sampling Technique (SMOTE) or Adaptive Synthetic Sampling (ADASYN) for scenarios where fraud cases are extremely rare.

3. Real-time Fraud Detection:
Implement streaming data pipelines (using tools like Apache Kafka and Spark Streaming) for detecting fraud in real-time instead of batch predictions.

4. Ensemble Models:
Combine the strengths of multiple models (stacking, blending, voting) to build a more robust fraud detection system.

5. Anomaly Detection Algorithms:
Explore more sophisticated anomaly detection techniques such as One-Class SVMs, Variational Autoencoders (VAEs), and GAN-based models for fraud detection.

6. Deep Learning Enhancements:
Test more complex architectures like Bidirectional LSTM (BiLSTM) or Transformer-based models to capture sequential transaction patterns better.

7. Cost-Sensitive Learning:
Introduce cost-based metrics where the cost of missing a fraud (false negative) is much higher than wrongly flagging a transaction (false positive).

8. Deployment:
Deploy the trained model using Flask/FastAPI and integrate it with a dashboard (like Power BI, Streamlit, or Dash) for real-time monitoring and alerts.

### Conclusion:
This project demonstrates the implementation of multiple machine learning and deep learning models for credit card fraud detection.
Models like Random Forest, XGBoost, LSTM, Autoencoders, Isolation Forest, and DBSCAN were applied to detect fraudulent transactions effectively.
Among these, LSTM, Random Forest, and XGBoost performed exceptionally well, achieving high ROC-AUC scores and reliable precision-recall metrics.

The project highlights the critical importance of accurate fraud detection systems in today's digital economy, where cyber threats are increasingly sophisticated.
While the current results are promising, there remains ample scope for further improvements through model tuning, real-time system development, and the integration of more advanced techniques.

Overall, this work provides a strong foundation for building scalable and efficient fraud detection solutions that can significantly enhance the security of financial transactions.
