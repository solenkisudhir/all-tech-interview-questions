# Anti-Money Laundering using Machine Learning
When we talk about financial crime, money laundering is one of the biggest threats in the financial world. Money Laundering is one of the most famous ways to convert black money into white money. Although various financial institutions follow some acts and rules to prevent the activity of money laundering, in this technology era where everything is digital and being recorded by financial software, it isn't easy to prevent such activities in traditional ways. Hence, all financial institutions are adopting and equipping themselves with powerful technologies and analytical tools to combat money laundering.

![Anti-Money Laundering using Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/anti-money-laundering-using-machine-learning.jpg)

Machine Learning also plays a significant role in detecting money laundering activities in financial institutions and automatically restricts users from using their accounts until the issue is resolved. Machine learning employs various algorithms to identify money laundering activities and prevent them greatly. In this topic, "**_Anti Money Laundering using Machine Learning_**", we will learn how a machine learning model can help identify suspicious account activity and provide better support to the AML team. Hence, before starting this topic, we must know these terms like money laundering anti-money laundering (AML). So let's start with a quick introduction to money laundering, anti-money laundering, and then Anti-money laundering using machine learning models.

What is Money Laundering?
-------------------------

**_Money laundering is defined as converting a large amount of money obtained from illegal sources into origination from a legitimate source._**

In simple words, it is a process to convert black money into white money.

Source of Money Laundering
--------------------------

Money laundering can be done with many sources such as Black Salaries, Round tripping, smuggling, illegal weapons, Casinos, multiple cash withdrawn in high cash jurisdictions, etc.

What is Anti Money laundering?
------------------------------

**_Anti-money laundering is defined as the laws, regulations, and procedures followed by banking and financial institutions to prevent money laundering activities._**

It includes three stages as follows:

*   **Placement**: This is the step where money obtained from illegal sources puts into financial institutions for the first time.
*   **Layering**: In these steps, money launders make various layers by dividing money into multiple bank accounts to confuse the banking analysts and ML algorithms, so they cannot identify the actual source of laundering.
*   **Integration**: This is the final step in sending this layered money to the money launder's account.

Anti Money Laundering (AML) using Machine Learning Applications:
----------------------------------------------------------------

Machine Learning plays a significant role in preventing money laundering activities in financial industries. To prevent money laundering, it uses a supervised machine learning technique in which an ML model is trained with various types of data or trends to identify the alerts and suspicious transactions flagged by the internal banking system. These machine learning models help identify these suspicious transactions, sender, and beneficiary financial records, their pattern of making transactions using transaction history, etc.

Machine Learning algorithms help in AML and reduce human error to a great extent. Machine learning models use a few techniques to prevent money laundering.

Natural Language Processing (NLP) helps machines process human language and identify alerts, process mortgage loans, negative news screening, payments screening, etc. Further, these machine learning technologies help monitor various suspicious activities and transaction monitoring. ML teaches machines to detect and identify the transaction patterns, behavior, associated suspicious users/accounts, and classification of alerts based on their risk categories such as High risk, medium risk, and low risk. Further, it checks alerts, automatically clears some alerts and makes accounts fully operational based on their account behavior and required documents.

Machines can be taught to recognize, score, triage, enrich, close, or hibernate alerts. However, these processes are very complex for humans and time-consuming, but with the help of machine learning technologies, they become relatively easier than the classical approach. Natural Language Generation (NLG) helps fill Suspicious Activity Reports (SAR) and provides the narratives for the same. This way can reduce dependencies on human operators to perform routine tasks, reduce the total time it takes to triage alerts, and allow personnel to focus on more valuable and complex activities.

With the introduction of ML into AML TM alert triage, SAR conversion rates should improve from the current unacceptable rate of ~1% in the banking sector.

![Anti-Money Laundering using Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/anti-money-laundering-using-machine-learning2.jpg)

Why use Machine Learning in Anti Money laundering (AML)
-------------------------------------------------------

Machine Learning is widely used in the banking and finance industry, and AML is one of the best examples of using machine learning. There are a few important reasons that show machine learning plays a vital role as follows:

*   **Reduction of false positive in the AML process:**

Machine learning helps identify and detect 98% of the false positives in the AML process, while compliance teams estimate only 1% and 2% of AML alerts. In the AML process, some alerts are generated wrongly that affect the customer's account by putting some restrictions. However, these alerts should not be triggered on the user's account. Machine learning helps to reduce the rate of false positives by using semantic analysis and statistical analysis to identify the risk factors that lead to true positive results. Machine learning algorithms help in eliminating these false positives during the transactions monitoring process.

*   **Detecting the change in customer behavior**

Machine Learning teaches computers past transactions and their Profileprofile that helps detect the customer behavior. These machines first learn with old data and then analyze it according to the customer's transaction history. According to their transaction behavior/patterns, these machines detect all suspicious activities and associated users who all were associated with any suspicious activity in the past. Using traditional approaches of finding customer behavior is not accurate and time-consuming; Machine Learning technology has reduced the chances of human errors. Also, it reduces the investigation time by monitoring customer transactions using rule engines.

Hence, machine learning makes this process relatively faster because money launderers are generally one step ahead.

*   **Analysis of unstructured data and external data**

Banking and financial institutions analyze customer data such as KYC, screening, residence country, professions, politically exposed person (PEP) status, social status, etc., to check their behavior. These all are the main factors that affect the business of any financial institution. To reduce the financial risk, financial institutions use many external datasets such as LinkedIn, Bloomberg, BBL, Norkom, social networks, company houses, and other open-source data.

BBL and Norkom are the software that helps find matches or name search using external data and tells computers if any customer is associated with any fraud/suspicious activity, PEP, high-risk entity. Hence, The NLP replaces these classical approaches and helps to analyze this unstructured data and establish the connection.

Hence, Machine learning technologies help to analyze the unstructured data and external data in a significant manner in comparison to classical methods with greater accuracy.

*   **Robotic Process Automation (RPA) in AML and KYC**

RPA plays a significant role in the banking and finance sectors. So many banks are still adapting the RPA to automate their business process. Further, When RPA is combined with Machine Learning, it becomes more powerful. It provides intelligent automatic techniques in different banking operations such as Know your customer (KYC), transaction monitoring, screening, alert elimination, etc.

RPA with machine learning helps in the following ways:

*   It helps create a 360-degree view of customer data, including data duplication and reconciliation from the back-end.
*   It helps create and update customer profile data using external data sources.
*   It helps in alerts elimination using external and internal data. Further, it also supports the enhancement of customer data like periodic KYC, alerts, Profileprofile, risk status, customer information portfolio, and geolocation data.
*   It helps to perform account analysis of ultimate beneficial owners using external data sources.

Challenges to Machine Learning in AML
-------------------------------------

A few challenges have been identified while implementing Machine Learning in Anti-money laundering or other financial services.

These challenges include data quality management (poor data quality), profile refresh, lack of 360-degree view of the customer, insufficient knowledge of banking, finance, and AML process such as know your customer (KYC), limited regulatory appetite, lack of straightforward processes to follow for machine learning implementations.

*   **Data Quality management & Profileprofile refresh:**

Data quality management is one of the most important factors for implementing machine learning applications in AML. It is required for both monitoring as well as for analytics purposes. Lack of data traceability and data lineage is also found in both static and dynamic customer profile records. Static data can be like KYC documents, and dynamic data may be their incoming and outgoing transactions.

Sometimes, it is also found that a few alerts are generated wrongly on customer accounts, i.e., false-positive, but actually, they are not likely to be generated on the account. This may lead to various types of restrictions on customers' accounts and affect the entire business. These issues reduce the reoccurrence of noise or false positives on the user's account. Further, other techniques are also applicable instead of using these methods, such as large-scale, one-off data reconciliation or refresh exercises, etc. Many FIs have undertaken large and costly data remediation projects to improve data and have implemented frameworks to manage data quality during the last few years. Hence, financial specialists always find data quality a major issue. On the other hand, profile refresh can also be a significant solution for managing quality data. Relationship managers and back-end associates can use profile refresh within a certain duration by reaching out to customers and validating their documents.

*   **Lack of 360-degree view of the customer:**

This is another important issue in implementing ML applications in the AML process. Financial institutions never disclose their customer data to build a comprehensive network. Further, FIs do not cooperate on AML to build a 360-degree view of customers among regulatory agencies as this is a cost-consuming method. Instead of using the above, FIs supports to file suspicious activity reports with appropriate automatic narrations and submit to the regulator and share information securely between FIs and regulators using external datasets like KYC. Some acts and regulations are also formed by financial institutions, such as US Patriot Act 314 a, 314 b & PSD2. Furthermore, the UK treasury also helps share data via an Open Banking API/Open Banking Working Group.

*   **Limited knowledge of both banking and financial services and ML:**

Machine Learning is a very new technology in the market, and there are very few ML engineers and professionals in the industry. Further, a lack of knowledge in banking and financial operations has also been seen in analysts, leading to various major problems from start-ups and established vendors. This is one of the most common factors found while implementing machine learning in AML and other banking operations.

*   **Limited regulatory appetite:**

The regulators need an ideal ML model that includes all choices, limitations, and results in the documented format before implementing it in the AML process. ML algorithms do not allow results to be reproduced with a given input, but regulators expect the result to be reproduced while implementing in the AML process. Some regulators want intelligent and adaptive solutions for transaction monitoring that have become a complex scenario for ML learning applications.

*   **Lack of straightforward process:**

Machine learning is a very new technology, and it is even under development. Hence, there are a few established, straightforward processes to follow to implement it. Teaching systems to detect certain types of financial crime can be tricky without knowing what to look for. For example, how does one teach a system to recognize terrorist financing? There is a carousel process for fraud but nothing similar for terrorist financing (nothing that is, other than name matching against terrorist lists). While some of these problems are better suited to unsupervised learning, model validators should be sure about the desired outcomes.

Conclusion
----------

Anti-money laundering is a broad field in the banking and financial industry, and this is one of the most important key factors in preventing the illegal flow of money. Machine Learning plays a significant role in the AML process to get better results with greater efficiency and effectiveness. Although many financial institutions also adopt automation like Robotics Process Automation (RPA) in their business process, some belief in machine learning and artificial intelligence to run their business. However, robotics can train ML models, and ML models help robotics build strong decision-making (in the form of NLP) or reading (via optical character recognition).

* * *