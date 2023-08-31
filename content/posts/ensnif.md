---
title: "Ensnif"
date: 2020-09-21T00:54:53-04:00
slug: "ensnif"
description: "Privacy-preserving platform to host data"
keywords: ["spark", "airflow", "mpc", "fully homomorphic encryption",  "time-series", "go"]
draft: false
tags: ["spark", "airflow", "mpc", "fhe", time-series", "go", "SCALE-MAMBA", "hudi"]
math: false
toc: false
---

## Update

This project is no longer active.

## Motivation

As Andrew Trask from OpenMined puts it (and I'm paraphrasing here): there are cases where public information is found in private information. The canonical example given by Trask is of building a cancer detection model using patient data: the model itself is not representative of any specific person but the individual data point used to build the model is unique. This private information is collected under the explicit trust (sometimes implicit) that the information won't be divulged (through any means). The conventional methods to keep this promise involved anonymizing data. However, the industry has denounced the practice since there were multiple cases where participants' private information was divulged despite the dataset being anonymized. 

It stands that much good can come from using private information but it must be done judiciously; with a platform that preserves this privacy and also allows groups to collaborate, we expect more data being shared to build smarter and more robust models without giving up access to the data.

## Introduction

We're building a platform where public and private institutions can host their data and make it available to the general public without giving access to the underlying information.

## Where do existing solutions fit in?

The business models used by the existing companies above offer highly curated services for their customers, which allows them to focus on their customers’ needs with an incredible level of detail.

However, a large portion of the market is not being served by this business model because either our competitors don’t have the bandwidth or the price is not justifiable. 

We’re more interested in a general purpose data sharing platform - on the level of cities trying to make data available to social programs or small businesses making their sales data available to marketing services.

## High-level Overview

This document is hopefully one of many and I will try to detail as much of this process as I can. For now, I'll talk about the overall flow.

The data marketplace allows data owners to share data. If an organization wants to serve their data for anyone, they encrypt their data using a fully homomorphic key and send the encrypted data to us. The nature of homomorphic schemes allows us to serve that data to help train another party's models. We also use concepts in differential privacy to ensure privacy as well. Refer to the section on differential privacy where I talk about how we can ensure that the underlying information is kept private. If you're curious, take a look at: [https://arxiv.org/pdf/1812.02292.pdf]. 
Organizations can also opt to share their data within a consortium of other organizations - for example, a grocery store can share their order data with the logistics team at a delivery company. We would use SCALE-MAMBA, a MPC library, to facilitate sharing between organizations. 

In any case, once the user decides to pick a dataset, we take the user to an airflow page where they can submit their git repo to run the code within the repo. 
Airflow is used to manage the workflows - we picked Airflow because of its rich user interface and reasonable friendly Kubernetes operator. Airflow is used to start a Spark job which will run and give the user their results. A similar workflow exists for a group of organizations in that all of the organizations see a separate airflow dashboard and they can submit their own programs. The results are sent back to the users who can decrypt the results to see it in plaintext.




