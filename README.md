# AI Psychological Counseling Agent Portfolio

## 1. Project Overview

### Project Title

**AI Psychological Counseling Agent for Young Adults in Their 20s and 30s**

### One-Line Description

An AI-powered psychological counseling and coaching service that analyzes a user's concerns, counseling intent, emotional state, possible causes, and safety risks, then provides personalized responses, emotional reflection reports, and professional counseling resources.

### Background

Young adults in their 20s and 30s experience various psychological challenges related to academic pressure, employment, relationships, sleep, and stress. However, practical and emotional barriers often make it difficult for them to access professional counseling services.

This project was designed to provide an accessible environment where users can express their concerns without pressure. The AI then analyzes their input in a structured manner and offers initial emotional support and practical, manageable actions.

### Core Objectives

* Provide an AI counseling interface that is accessible 24/7
* Analyze user input through an **Agent Pipeline** rather than generating a simple chatbot response
* Structure counseling intent, emotional state, possible causes, and safety risks
* Detect crisis-related expressions and provide professional counseling resources and emergency contact information
* Generate a structured **Emotional Reflection Report** based on the counseling flow
* Use datasets as contextual evidence for agent decision-making rather than as direct response templates

---

## 2. Key Features

| Feature                           | Description                                                                                        |
| --------------------------------- | -------------------------------------------------------------------------------------------------- |
| AI Counseling Chat                | Generates counseling responses based on the user's concerns                                        |
| Agent Pipeline Analysis           | Analyzes the counseling flow through Intent, Emotion, Cause, Safety, Decision, and Response Agents |
| Emotional Reflection Report       | Summarizes the user's current state, possible causes, small actions, and next steps                |
| Professional Counseling Resources | Provides access to counseling organizations when risks are detected or support is requested        |
| Anonymous Mode                    | Allows users to receive support without entering personal information                              |
| Consent-Based Record Storage      | Stores structured counseling data only when the user provides consent                              |
| Previous Session Recovery         | Restores previous counseling summaries and reports for authenticated users                         |

---

## 3. System Architecture

### Overall Workflow

```text
User Input
    ↓
Input Preprocessing
    ↓
Safety Agent
    ↓
Intent Agent
    ↓
Emotion Agent
    ↓
Hint Retrieval
    ↓
Cause Agent
    ↓
Decision Agent
    ↓
Gemini API / Response Agent
    ↓
Response Polish Layer
    ↓
Final Counseling Response
Emotional Reflection Report
Professional Counseling Resources
```

### Agent Responsibilities

| Agent                       | Input                                         | Responsibility                                                                   | Output                                       |
| --------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------- |
| Safety Agent                | User message                                  | Detects crisis-related expressions such as suicide or self-harm                  | Risk level                                   |
| Intent Agent                | Message and conversation context              | Classifies concerns such as academic pressure, sleep issues, anxiety, and stress | `primary_intent`                             |
| Emotion Agent               | Message and user-selected check values        | Identifies anxiety, stress, sleep quality, and energy level                      | `emotional_state`                            |
| Hint Retrieval              | Curated datasets                              | Retrieves counseling, empathy, and wellness-related contextual hints             | Context hints                                |
| Cause Agent                 | Intent, hints, and previous conversation flow | Infers possible causes of the user's current state                               | `selected_cause`                             |
| Decision Agent              | Risk level, cause, and emotional state        | Determines the response strategy                                                 | Empathy, follow-up question, or small action |
| Gemini API / Response Agent | Agent decisions and user message              | Generates a natural-language counseling response                                 | Counseling response                          |
| Response Polish Layer       | Original user message and generated response  | Applies safety and tone adjustments                                              | Safety-adjusted final response               |

---

## 4. Dataset Design and Usage

### Original Dataset Overview

| Category        | Number of Records | Purpose                                                        |
| --------------- | ----------------: | -------------------------------------------------------------- |
| Counseling Data |           204,962 | Counseling intent, possible causes, and response strategy      |
| Empathy Data    |           204,356 | Emotional reflection, empathetic language, and tone adjustment |
| Wellness Data   |             1,216 | Sleep, stress, daily routines, and small practical actions     |
| Total           |           410,534 | Original candidate pool for agent decision-making              |

### Dataset Usage Principles

The datasets in this project are not used to directly copy and output predefined answers.

Instead, they are used as **Context and Knowledge** that help the agents determine:

* Counseling intent
* Emotional state
* Possible causes
* Appropriate response strategies
* Small and manageable actions

### Why Dataset Reduction Was Necessary

Using all 410,534 records could cause several problems:

* Slower response times due to an excessive number of retrieval candidates
* Increased likelihood of retrieving hints that do not match the current context
* Higher prompt length, API cost, and latency
* Timeout or infinite-loading risks during live demonstrations
* Increased possibility of retrieving inappropriate or unsafe counseling hints

### Dataset Versions

| Dataset     | Counseling | Empathy | Wellness | Total | Purpose                                    |
| ----------- | ---------: | ------: | -------: | ----: | ------------------------------------------ |
| Small       |      1,400 |   1,400 |      700 | 3,500 | Functional validation                      |
| Balanced    |      2,231 |   2,231 |    1,216 | 5,678 | Dataset balance                            |
| Broad       |      3,125 |   3,125 |    1,216 | 7,466 | Wider topic coverage                       |
| Recommended |      2,068 |   2,067 |    1,216 | 5,351 | Final production and demonstration dataset |

### Small Dataset

The Small dataset was created for rapid functional validation.

It was used to confirm that core functions such as authentication, report generation, crisis responses, and professional counseling resources were working correctly.

However, its counseling-topic coverage was too limited for use as the final decision-support dataset.

### Balanced Dataset

The original counseling and empathy datasets each contained more than 200,000 records, while the wellness dataset contained only 1,216 records.

The Balanced dataset was created to reduce this imbalance. All wellness records were retained, while the counseling and empathy datasets were reduced to similar sizes.

This structure allowed counseling analysis, empathy, and practical wellness suggestions to contribute more evenly.

### Broad Dataset

The Broad dataset was designed to support a wider variety of counseling topics, including:

* Academic pressure
* Anxiety
* Sleep problems
* Relationship concerns
* Stress

Although it offered broader coverage, its larger retrieval pool created additional challenges in response speed and demonstration stability.

### Recommended Dataset

The Recommended dataset was selected for final execution and live presentation.

All wellness records were retained because they were limited in number and important for generating small practical actions. The counseling and empathy datasets were reduced to approximately 2,000 records each to achieve a balance between quality, coverage, and speed.

### Final Selection Criteria

| Evaluation Criterion       | Decision                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------ |
| Counseling Topic Diversity | Provides broader topic coverage than the Small dataset                               |
| Safety                     | Contains less noise and fewer potentially inappropriate hints than the Broad dataset |
| Response Speed             | Better suited for real-time interaction than the full or Broad datasets              |
| Demonstration Stability    | Appropriate for a stable five-minute live demonstration                              |

The final Recommended dataset contains 5,351 records. It offers greater diversity than the Small dataset, better stability than the Broad dataset, and a lighter structure than the Balanced dataset.

---

## 5. Gemini API Integration

Gemini API is not used as a standalone chatbot in this project.

Instead, it operates as the final **Response Agent** within the Agent Pipeline.

### Response Generation Flow

```text
User Message
    ↓
Agent Decision Results
    ↓
Natural-Language Response Generated by Gemini API
    ↓
Safety and Tone Adjustment through the Response Polish Layer
    ↓
Final Response
```

### Purpose of Gemini API

* Convert structured agent decisions into natural counseling responses
* Generate empathetic language appropriate to the user's context
* Present small practical actions in a natural and supportive manner
* Reflect retrieved dataset hints and the current conversation flow

### Safety Adjustment Logic

Crisis handling does not rely solely on Gemini API responses.

A code-level safety adjustment layer provides Korean emergency and counseling contact information when necessary.

| Situation                   | Contact Information             |
| --------------------------- | ------------------------------- |
| Suicide or self-harm risk   | Suicide Prevention Hotline: 109 |
| Youth counseling            | Youth Counseling Hotline: 1388  |
| Emergency police assistance | 112                             |
| Medical emergency           | 119                             |

---

## 6. Emotional Reflection Report

The Emotional Reflection Report organizes the user's counseling flow into a structured summary.

Rather than storing the entire conversation, the system stores only the essential counseling information as a structured snapshot.

### Stored Information

* Risk level
* Primary counseling intent
* Emotional state
* Possible causes
* Suggested small actions
* Current action status
* Summary of the counseling flow

### Design Objectives

* Help users understand their current emotional state at a glance
* Preserve essential context for future counseling sessions
* Reduce privacy risks by minimizing storage of original conversation text
* Store structured snapshots only when the user has provided consent

---

## 7. Privacy and Safety Design

### Privacy Principles

* Anonymous mode is available
* Users can choose whether to allow counseling records to be stored
* Full original conversations are not stored by default
* Counseling flows are stored as structured snapshots
* Users who do not provide consent can use the service on a session-only basis

### Safety Measures

* Crisis-expression detection
* Korea-specific emergency contact guidance
* Professional counseling resource section
* Post-processing of Gemini API responses through a safety adjustment layer
* Clear acknowledgment that the AI does not replace professional mental health services

---

## 8. Technology Stack

| Area                 | Technology                                                    |
| -------------------- | ------------------------------------------------------------- |
| Frontend and Demo UI | Gradio                                                        |
| Backend Logic        | Python                                                        |
| Large Language Model | Gemini API                                                    |
| Data Processing      | Preprocessed JSONL datasets                                   |
| Report Cache         | JSON-based local cache                                        |
| Agent Pipeline       | Intent, Emotion, Cause, Safety, Decision, and Response Agents |
| Deployment and Demo  | Localhost-based demonstration                                 |

---

## 9. Problems Solved During Development

### 1. Response Delays When Using the Full Dataset

The original dataset contained 410,534 records. Using the entire dataset created an excessively large retrieval pool and increased response latency.

To solve this issue, multiple purpose-specific datasets were created. The Recommended dataset containing 5,351 records was ultimately selected for final execution.

### 2. Gemini API Response Delays

External API calls can be affected by network conditions and the execution environment.

To improve reliability, local safety-adjusted responses were designed for crisis-related inputs and important demonstration scenarios.

### 3. Infinite Loading During Report Generation

The Emotional Reflection Report could appear to load indefinitely when external calls or restoration logic took too long.

This issue was addressed by implementing a local report-generation process based on structured counseling snapshots.

### 4. Professional Counseling Resource UX

Displaying the professional counseling button only during a crisis created unstable UI update behavior.

The interface was improved by keeping professional counseling resources available at all times, allowing users to access them whenever needed.

---

## 10. Demonstration Flow

### Recommended Demonstration Input

```text
I feel overwhelmed because of my studies.
```

### Demonstration Steps

1. The user enters a short description of their concern
2. The Agent Pipeline analyzes counseling intent and possible causes
3. The AI generates a personalized counseling response
4. The user reviews the Emotional Reflection Report
5. The user opens the professional counseling resources section

### Key Points Demonstrated

* Short user inputs can be classified into counseling intents
* Emotional states and possible causes are analyzed structurally
* Datasets are used as decision-support hints rather than copied answers
* Counseling responses, reports, and professional resources are connected within a single service flow

---

## 11. Project Outcomes

### Implementation Outcomes

* Developed an AI counseling chat interface
* Designed an Agent Pipeline-based counseling analysis structure
* Implemented Gemini API-based response generation
* Created multiple reduced datasets and established final selection criteria
* Implemented the Emotional Reflection Report
* Developed a professional counseling resource flow
* Added anonymous mode and consent-based record storage

### Design Outcomes

* Designed an agent-based counseling workflow rather than a simple chatbot
* Used datasets as Context and Knowledge rather than direct response sources
* Implemented safety adjustment logic appropriate for a psychological counseling service
* Optimized dataset size and execution structure for live demonstration stability

---

## 12. Limitations and Future Improvements

### Limitations

* The service cannot replace professional counseling or clinical treatment
* Gemini API response speed may vary depending on external conditions
* The current implementation is primarily designed as a local demonstration
* Crisis-risk detection includes keyword-based logic and requires further improvement
* Long-term personalization using historical user data is currently limited

### Future Improvements

* Introduce vector database-based retrieval
* Strengthen source-level metadata management for counseling datasets
* Improve crisis-risk classification models
* Integrate counseling organization APIs
* Analyze long-term emotional changes with user consent
* Optimize the interface for mobile environments
* Expand integration workflows with professional counselors

---

## 13. Portfolio Summary

This project is not a simple AI chatbot. It is an AI psychological counseling service that analyzes user input through an Agent Pipeline and uses curated datasets as decision-support context.

The system provides:

* Structured counseling analysis
* Personalized AI responses
* Emotional Reflection Reports
* Professional counseling resources
* Crisis-related safety guidance

Rather than using all 410,534 original records directly, the datasets were reorganized into Small, Balanced, Broad, and Recommended versions.

The final Recommended dataset contains 5,351 records and was selected based on counseling-topic diversity, safety, real-time response speed, and stability during a five-minute live demonstration.

Gemini API is used as the final Response Agent. Crisis-related expressions are separately processed through the Safety Agent and Response Polish Layer, which provide Korea-specific emergency guidance when necessary.

Through this structure, the project demonstrates an agent-based AI counseling service that combines structured decision-making, dataset-based contextual reasoning, natural-language generation, privacy protection, and safety-oriented response design.

---

## Disclaimer

This project is intended for educational, research, and portfolio demonstration purposes.

It does not provide medical diagnoses, clinical treatment, or professional mental health counseling. Users experiencing an immediate crisis should contact local emergency services or qualified mental health professionals.
