[MS Flow Actions]( https://learn.microsoft.com/en-us/power-automate/desktop-flows/actions-reference/sharepoint)

Power Automate is a powerful tool for automating workflows across different apps and services. Here are some common actions you can perform with Power Automate, along with brief descriptions:

1. **Send an Email**: This action allows you to send an email from your connected email account. You can specify the recipient, subject, body, and other email properties.

2. **Create a SharePoint Item**: With this action, you can create a new item in a SharePoint list or library. You can define the fields and their values for the new item.

3. **Post a Message to Microsoft Teams**: This action lets you post a message to a specified channel in a Microsoft Teams workspace. You can include text, links, and other content in the message.

4. **Create a File in OneDrive or SharePoint**: Use this action to create a new file in a OneDrive folder or SharePoint document library. You can specify the file name, content, and other properties.

5. **Trigger a Power BI Data Refresh**: With this action, you can trigger a data refresh for a Power BI dataset. This is useful for keeping your Power BI reports up-to-date with the latest data.

6. **Create a Task in Microsoft Planner**: This action allows you to create a new task in a Microsoft Planner plan. You can specify the task title, due date, assigned user, and other details.

7. **Send a Notification**: Use this action to send a notification to a user's mobile device using the Power Automate mobile app. Notifications can include text and links.

8. **Update a Salesforce Record**: With this action, you can update an existing record in Salesforce. You can specify the record type, fields to update, and new values.

9. **Start Approval Process**: This action initiates an approval process for a specified item or document. Approvers can review the item and either approve or reject it.

10. **Create a Calendar Event**: Use this action to create a new event in a calendar service like Outlook or Google Calendar. You can specify the event title, date, time, location, and other details.

Certainly! Let's delve deeper into some of the Power Automate actions with more detailed explanations:

1. **Send an Email**:
   - This action allows you to send emails programmatically, integrating with various email services like Outlook, Gmail, or SMTP. 
   - You can dynamically populate email fields such as recipient, subject, body, and attachments using data from previous steps in your flow.
   - Additional options may include setting importance levels, adding HTML content, or specifying the sender's name and email address.

2. **Create a SharePoint Item**:
   - SharePoint is a collaboration platform that allows organizations to store, share, and manage documents and data.
   - With this action, you can create new items in SharePoint lists or libraries, specifying values for various fields.
   - Fields could include text, numbers, dates, choices, or even lookups to other SharePoint lists.
   - You can also handle error cases, like if the item creation fails due to validation rules or permissions.

3. **Post a Message to Microsoft Teams**:
   - Microsoft Teams is a hub for teamwork, providing chat, meetings, file sharing, and integration with other Microsoft 365 apps.
   - This action lets you send messages to Teams channels, facilitating communication and collaboration within your organization.
   - You can format messages using Markdown, include @mentions to notify specific users or groups, and even attach files or images.
   - It's useful for automating notifications, status updates, or sharing important information with your team members.

4. **Create a File in OneDrive or SharePoint**:
   - OneDrive and SharePoint are cloud storage services by Microsoft, offering file sharing, collaboration, and versioning capabilities.
   - This action allows you to create new files programmatically, specifying file names, content, and metadata.
   - You can create Word documents, Excel spreadsheets, PowerPoint presentations, or any other file type supported by OneDrive or SharePoint.
   - It's useful for generating reports, storing data exports, or dynamically creating documents based on templates.

5. **Trigger a Power BI Data Refresh**:
   - Power BI is a business analytics service by Microsoft that provides interactive data visualization and business intelligence capabilities.
   - This action triggers a dataset refresh in Power BI, ensuring that your reports and dashboards reflect the most recent data.
   - You can specify the dataset to refresh and optionally wait for the refresh to complete before proceeding with subsequent steps in your flow.
   - It's useful for automating data pipelines, ensuring data accuracy, and keeping your analytics up-to-date.

Certainly, let's continue exploring more Power Automate actions in detail:

6. **Create a Task in Microsoft Planner**:
   - Microsoft Planner is a task management tool that allows teams to organize and track their work using boards, tasks, and buckets.
   - This action enables you to create new tasks in a specific Planner plan, providing details such as task title, due date, assigned user, and description.
   - You can also set additional properties like priority, labels, or checklist items to further organize and categorize the task.
   - It's useful for automating task creation based on triggers from other systems or processes, ensuring that work items are efficiently managed and tracked.

7. **Send a Notification**:
   - This action sends a push notification to a user's mobile device using the Power Automate mobile app.
   - Notifications can contain text, links, and actions, allowing users to quickly respond or take necessary actions.
   - You can customize the notification message and format to provide relevant information and context to the recipient.
   - It's useful for sending alerts, reminders, or notifications about important events or updates to users, ensuring timely communication and action.

8. **Update a Salesforce Record**:
   - Salesforce is a cloud-based customer relationship management (CRM) platform that helps businesses manage their sales, marketing, and customer support processes.
   - With this action, you can update existing records in Salesforce objects such as leads, accounts, contacts, or custom objects.
   - You specify the record ID and the fields to be updated along with their new values.
   - It's useful for keeping Salesforce data synchronized with changes in other systems or automating updates based on specific criteria or events.

9. **Start Approval Process**:
   - This action initiates an approval process for a specified item or document, routing it to one or more approvers for review and decision.
   - Approvers can approve, reject, or request changes to the item, and you can define custom approval conditions and escalation rules.
   - You can track the status of approvals, receive notifications on approval outcomes, and take actions based on the approval result.
   - It's useful for automating approval workflows such as document reviews, expense requests, or leave applications, ensuring proper authorization and compliance with business rules.

10. **Create a Calendar Event**:
    - This action creates a new event in a calendar service like Outlook, Google Calendar, or Microsoft 365 Calendar.
    - You specify details such as event title, start and end dates/times, location, description, attendees, and reminders.
    - Calendar events can be dynamically generated based on triggers from other systems or scheduled processes.
    - It's useful for automating event scheduling, reminders, or coordination across teams or individuals, helping to manage time effectively and avoid scheduling conflicts.

Certainly, let's continue exploring additional Power Automate actions:

11. **Create a SharePoint List Item**:
    - Similar to creating a SharePoint item, this action specifically targets SharePoint lists. It allows you to add new items to a list, specifying values for different columns.
    - You can customize the fields to be populated and handle any errors that may occur during the item creation process.

12. **Send a Text Message (SMS)**:
    - This action enables you to send SMS messages to mobile phone numbers using supported SMS providers.
    - You can customize the message content, specify the recipient's phone number, and include dynamic content using data from previous steps in the flow.
    - It's useful for sending notifications, alerts, or reminders via SMS, especially when immediate attention or confirmation is required.

13. **Create a Task in Asana or Trello**:
    - Asana and Trello are popular task management tools used for project planning and collaboration.
    - This action allows you to create tasks in Asana or Trello boards, providing details such as task name, description, due date, assignee, and other relevant properties.
    - It's useful for integrating Power Automate with project management workflows, automating task creation based on triggers or events from other systems.

14. **Update a Microsoft Excel or Google Sheets Row**:
    - Excel and Google Sheets are widely used spreadsheet applications for data analysis and reporting.
    - With this action, you can update specific rows in Excel or Google Sheets workbooks, modifying cell values based on predefined criteria.
    - You specify the worksheet, row, and column values to be updated, as well as the new data to replace existing values.
    - It's useful for automating data updates in spreadsheets, ensuring accuracy and consistency across multiple systems and processes.

15. **Create a File in Dropbox or Box**:
    - Dropbox and Box are cloud storage services that provide file hosting, sharing, and collaboration features.
    - This action allows you to create new files in Dropbox or Box folders, specifying file names, content, and optional metadata.
    - You can dynamically generate file content or use predefined templates to streamline file creation processes.
    - It's useful for automating file generation and storage tasks, such as document generation, report exports, or file backups.

Of course, here are more Power Automate actions for your consideration:

16. **Create a Record in Dynamics 365**:
    - Dynamics 365 is a suite of intelligent business applications by Microsoft, including CRM and ERP solutions.
    - This action allows you to create new records in Dynamics 365 entities such as leads, accounts, contacts, opportunities, or custom entities.
    - You can specify the fields to populate in the new record, ensuring data accuracy and completeness.
    - It's useful for automating data entry processes, lead generation, or customer onboarding workflows.

17. **Send a Notification to Microsoft Teams Channel with Options**:
    - This action sends a message to a Microsoft Teams channel, similar to the "Post a Message to Microsoft Teams" action.
    - However, it allows you to include actionable options in the message, such as buttons or quick replies, for users to interact with.
    - You can define custom actions associated with each option, enabling users to perform actions directly from the Teams channel.
    - It's useful for implementing interactive notifications, surveys, or approval requests within Teams channels.

18. **Create a Record in ServiceNow**:
    - ServiceNow is a cloud-based platform that provides IT service management (ITSM), IT operations management (ITOM), and IT business management (ITBM) solutions.
    - With this action, you can create new records in ServiceNow tables, such as incident, change request, or task.
    - You specify the fields to populate in the new record, ensuring data consistency and adherence to ServiceNow workflows.
    - It's useful for automating IT processes, service ticketing, or incident management tasks.

19. **Send Approval Email**:
    - This action sends an approval request email to one or more approvers, allowing them to review and respond to the request directly from their email inbox.
    - You can customize the email content, including details about the request, instructions for approval or rejection, and any attachments or links.
    - Approvers can respond to the approval request by clicking on predefined buttons or links in the email, simplifying the approval process.
    - It's useful for implementing email-based approval workflows, such as document approvals, purchase requests, or expense reimbursements.

20. **Create a Record in Google Sheets**:
    - Google Sheets is a cloud-based spreadsheet application by Google, similar to Microsoft Excel.
    - This action allows you to create new rows in Google Sheets spreadsheets, specifying values for different columns.
    - You specify the spreadsheet and worksheet to insert the new row, as well as the data to populate in each column.
    - It's useful for automating data logging, form submissions, or data synchronization between different systems and platforms.

Absolutely, let's dive into more Power Automate actions:

21. **Create a Record in SQL Server**:
    - SQL Server is a relational database management system (RDBMS) developed by Microsoft.
    - This action enables you to insert new records into a SQL Server database table.
    - You specify the database connection, table name, and the values to be inserted into each column.
    - It's useful for integrating Power Automate with on-premises or cloud-based SQL Server databases, automating data entry tasks, or logging information.

22. **Post a Tweet**:
    - This action allows you to post a tweet to Twitter programmatically.
    - You can compose the tweet message dynamically using data from previous steps in the flow.
    - Twitter handles, hashtags, and links can be included in the tweet message.
    - It's useful for automating social media marketing campaigns, sharing updates, or broadcasting notifications on Twitter.

23. **Create a Record in Salesforce**:
    - Salesforce is a cloud-based CRM platform that helps organizations manage customer relationships, sales, and marketing processes.
    - With this action, you can create new records in Salesforce objects such as leads, accounts, contacts, opportunities, or custom objects.
    - You specify the object type and the field values to be populated in the new record.
    - It's useful for integrating Power Automate with Salesforce workflows, automating data entry tasks, or synchronizing information between systems.

24. **Trigger a Power Automate Flow from Another Flow**:
    - This action allows you to trigger one Power Automate flow from another flow.
    - You can pass inputs to the triggered flow, enabling dynamic and interconnected automation scenarios.
    - It's useful for orchestrating complex workflows, splitting tasks into smaller, reusable flows, or implementing modular automation architectures.

25. **Send Approval Request to Slack**:
    - Slack is a collaboration hub that brings team communication and workflows together in one place.
    - This action sends an approval request message to a Slack channel or user, allowing them to approve or reject the request directly within Slack.
    - You specify the message content, options for approval or rejection, and any additional details or attachments.
    - It's useful for integrating approval workflows with Slack channels, enabling seamless communication and decision-making within teams.

Certainly, here are more Power Automate actions to explore:

26. **Create a Record in SharePoint Online**:
    - This action allows you to create new items in SharePoint Online lists or libraries.
    - You can specify the list or library, along with the column values for the new item.
    - It's useful for automating data entry tasks, such as logging form submissions or recording events, directly into SharePoint Online.

27. **Post a Message to Slack**:
    - Slack is a popular collaboration platform for teams, offering messaging, file sharing, and integration with other tools.
    - This action enables you to post messages to Slack channels or direct messages (DMs).
    - You can include text, links, emojis, and even attachments in your Slack messages.
    - It's useful for automating notifications, sharing updates, or triggering discussions within Slack channels.

28. **Create a Record in Microsoft Dataverse**:
    - Microsoft Dataverse, formerly known as the Common Data Service (CDS), is a cloud-based data storage service by Microsoft.
    - This action allows you to create new records in Dataverse tables/entities, specifying field values for each record.
    - You can define relationships between entities and ensure data integrity and consistency.
    - It's useful for building custom business applications, managing structured data, and integrating with other Microsoft Power Platform services.

29. **Create a File in Azure Blob Storage**:
    - Azure Blob Storage is a cloud-based object storage service by Microsoft Azure.
    - This action enables you to create new files in Azure Blob Storage containers.
    - You specify the container name, file name, content, and optional metadata for the new file.
    - It's useful for automating file storage and management tasks in Azure, such as archiving data, storing backups, or processing file uploads.

30. **Create a Record in Oracle Database**:
    - Oracle Database is a relational database management system (RDBMS) developed by Oracle Corporation.
    - This action allows you to insert new records into Oracle Database tables.
    - You specify the database connection details, table name, and the values to be inserted into each column.
    - It's useful for integrating Power Automate with Oracle databases, automating data entry tasks, or synchronizing information between systems.

Certainly! Here are some common Power Automate actions along with detailed explanations:

1. **Send an Email**:
   - This action allows you to send an email to one or more recipients.
   - You can specify the recipient's email address, subject line, and email body.
   - Additionally, you can customize the email by including attachments, setting the importance level, and formatting the body with HTML.

2. **Create a SharePoint Item**:
   - SharePoint is a collaboration platform for document management and sharing.
   - This action enables you to create a new item in a SharePoint list or library.
   - You specify the list or library where the item will be created and provide values for each field in the item.

3. **Post a Message to Microsoft Teams**:
   - Microsoft Teams is a chat-based collaboration platform.
   - This action allows you to post a message to a specified Teams channel.
   - You can include text, links, and formatting in the message, making it useful for team communication and updates.

4. **Create a File in OneDrive or SharePoint**:
   - OneDrive and SharePoint are file storage and sharing services.
   - This action lets you create a new file in a specified OneDrive folder or SharePoint document library.
   - You provide the file name and content, and optionally specify the folder location and file properties.

5. **Trigger a Power BI Data Refresh**:
   - Power BI is a business analytics service for creating interactive reports and dashboards.
   - This action triggers a data refresh for a Power BI dataset.
   - You specify the dataset to refresh, ensuring that your reports reflect the latest data from your data sources.

6. **Create a Task in Microsoft Planner**:
   - Microsoft Planner is a task management tool for organizing work and team collaboration.
   - This action creates a new task in a specified Planner plan.
   - You provide details such as the task title, due date, assignee, and description.

7. **Send a Notification**:
   - This action sends a notification to a specified user or group.
   - Notifications can be sent through various channels, such as email or mobile push notifications.
   - You can customize the notification message and include dynamic content.

8. **Update a Salesforce Record**:
   - Salesforce is a customer relationship management (CRM) platform.
   - This action updates an existing record in Salesforce, such as an account, lead, or opportunity.
   - You specify the record ID and provide values for the fields you want to update.

9. **Start Approval Process**:
   - This action initiates an approval process for a specified item or document.
   - Approvers receive a notification and can approve or reject the request.
   - You define the approval criteria and actions to take based on the outcome.

10. **Create a Calendar Event**:
    - This action creates a new event in a calendar service such as Outlook or Google Calendar.
    - You specify details such as the event title, start and end times, location, and attendees.

Certainly! Let's continue with more common Power Automate actions and their details:

11. **Send a Text Message (SMS)**:
    - This action allows you to send SMS messages to mobile phone numbers.
    - You provide the recipient's phone number and the message content.
    - Some SMS providers may require authentication or API keys for integration.

12. **Create a Record in Google Sheets**:
    - Google Sheets is a cloud-based spreadsheet application.
    - This action creates a new row in a specified Google Sheets spreadsheet.
    - You specify the spreadsheet, worksheet, and values for each column in the new row.
    - It's useful for logging data, recording form submissions, or maintaining records in Google Sheets.

13. **Update a Microsoft Excel or Google Sheets Row**:
    - This action updates an existing row in a Microsoft Excel or Google Sheets spreadsheet.
    - You specify the spreadsheet, worksheet, and the row to update based on a unique identifier (e.g., row number or key).
    - You provide the new values for each column in the row to be updated.

14. **Create a Record in Dynamics 365**:
    - Dynamics 365 is a suite of intelligent business applications by Microsoft.
    - This action creates a new record in a specified Dynamics 365 entity, such as leads, accounts, contacts, or opportunities.
    - You specify the entity type and provide values for the fields in the new record.

15. **Post a Tweet**:
    - This action allows you to post a tweet to Twitter.
    - You compose the tweet message and can include hashtags, mentions, and URLs.
    - Twitter APIs may have rate limits and authentication requirements for posting tweets.

16. **Create a File in Dropbox or Box**:
    - Dropbox and Box are cloud-based file storage services.
    - This action creates a new file in a specified Dropbox folder or Box directory.
    - You provide the file name, content, and specify the destination folder.

17. **Create a Record in Salesforce**:
    - This action creates a new record in Salesforce, similar to the "Update a Salesforce Record" action.
    - You specify the object type (e.g., Account, Contact, Opportunity) and provide values for the fields in the new record.

18. **Send an Adaptive Card to Microsoft Teams**:
    - Adaptive Cards are interactive user interface elements used in Microsoft Teams and other Microsoft services.
    - This action sends an adaptive card to a specified Teams channel or user.
    - You define the card's layout, content, and actions, allowing users to interact with the card directly in Teams.

19. **Create a Record in ServiceNow**:
    - ServiceNow is a cloud-based platform for IT service management (ITSM) and business process automation.
    - This action creates a new record in ServiceNow, such as an incident, change request, or task.
    - You specify the table and provide values for the fields in the new record.

20. **Create a File in Azure Blob Storage**:
    - Azure Blob Storage is a cloud-based object storage service by Microsoft Azure.
    - This action creates a new file in a specified Azure Blob Storage container.
    - You provide the file name, content, and specify the destination container.

Power Automate provides several actions for making API requests to interact with external services or custom endpoints. Here are some common API request actions along with details:

1. **HTTP - HTTP**:
   - This action enables you to make HTTP requests to any publicly accessible web service or API endpoint.
   - You specify the HTTP method (GET, POST, PUT, DELETE, etc.), URL, headers, and body content.
   - It's versatile and can be used to interact with RESTful APIs, webhooks, or any HTTP-based service.
   - You can handle responses and errors using subsequent actions in your flow.

2. **HTTP with Azure AD - HTTP**:
   - Similar to the basic HTTP action, this action allows you to make HTTP requests but with Azure Active Directory (Azure AD) authentication.
   - You can authenticate using Azure AD credentials or a service principal.
   - It's useful for accessing resources protected by Azure AD, such as Microsoft Graph API or custom APIs registered in Azure AD.

3. **HTTP - HTTP + Swagger/Swagger with Azure AD**:
   - These actions enable you to make HTTP requests based on Swagger/OpenAPI specifications.
   - You import a Swagger/OpenAPI definition file, and Power Automate generates actions corresponding to the API endpoints defined in the specification.
   - The actions provide pre-defined inputs and outputs based on the API schema, simplifying the integration process.
   - You can authenticate using Azure AD if the API requires authentication.

4. **HTTP - HTTP + OpenAPI/Swagger (from URL)**:
   - Similar to the previous actions, this action allows you to make HTTP requests based on Swagger/OpenAPI specifications.
   - Instead of importing a local file, you provide the URL of the Swagger/OpenAPI definition file.
   - Power Automate retrieves the specification from the URL and generates actions for the API endpoints accordingly.
   - It's useful for integrating with APIs that expose Swagger/OpenAPI documentation online.

5. **Send an HTTP request to SharePoint**:
   - This action is specific to SharePoint and allows you to make HTTP requests to SharePoint REST API endpoints.
   - You can perform operations such as querying lists, creating items, updating properties, and more.
   - It simplifies interactions with SharePoint data and customization of SharePoint workflows.

6. **Send an HTTP request to Azure Resource Manager**:
   - This action enables you to interact with Azure resources using Azure Resource Manager (ARM) REST API.
   - You can perform operations like creating, updating, deleting resources, or managing resource groups and subscriptions.
   - It's useful for automating Azure resource provisioning, management, and monitoring tasks.

7. **HTTP - HTTP + Swagger/Swagger with Azure AD (preview)**:
   - This is a preview feature that combines the capabilities of the HTTP action with Swagger/OpenAPI specifications and Azure AD authentication.
   - It offers a unified experience for integrating with APIs defined using Swagger/OpenAPI and secured with Azure AD.
   - You import the API definition file, authenticate with Azure AD, and access the API endpoints seamlessly.

These Power Automate API request actions empower users to interact with a wide range of external services, APIs, and Azure resources, enabling automation of diverse workflows and integration scenarios.
