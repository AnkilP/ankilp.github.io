---
title: Anti-fragility in (Distributed) Systems
author: Ankil Patel
pubDatetime: 2023-08-30
slug: antifragile
featured: true
draft: false
tags:
  - antifragile
  - distributed systems
  - optimizations
  - reinforcement learning
description: The distributed system world is up for a paradigm shift
---

## Motivation

The holy grail isn't in resilient systems - it's in anti fragile systems (from the book by Taleb). Anti-fragile systems are able to learn and become better with new stressors. How do we build antifragile systems? I believe that framing complex systems problems as pure optimization problems lets us create AI agents that can adapt to new information and learn more efficiently.

A part of me wondered how many of the system-level issues we saw came from insufficient designs, changing environments that were no longer suitable to the implementation, terrible metrics (or lack thereof) and intuitive decision-making vs data-driven processes. Unfortunately, most of these problems are endemic to the entire industry, partly because of inertia, lack of infrastructure and de-prioritization because these issues aren’t visible to customers. However, the drivers behind decision-making are still cost, efficiency and productivity so I can see how the solutions outlined in this post address KPIs that businesses are measuring.

I understand why it’s been designed this way: classical control theory has rigorously proven theorems and established methods of achieving stability and correctness - hence why kubernetes is still able to use controllers invented in the 1900s to scale to ’large’ workloads. When I asked my reinforcement learning (RL) professor what the difference was between reinforcement learning and control theory, his response was that the two research groups just don’t meet despite solving very similar problems. Even Sutton and Barto in their seminal RL book talk about how control theory is a [precursor](https://ai.stackexchange.com/questions/11375/what-is-the-difference-between-reinforcement-learning-and-optimal-control) to RL but have diverged because of their differing objectives. It seems like [data-driven control theory](https://fab.cba.mit.edu/classes/865.21/topics/control/07_reinforcement_learning.html) is going to use more modern ML techniques and the line between these two fields gets more blurry. My preference for RL is not motivated by capability, because it seems that gap is getting smaller, but rather by simplicity of the model because we’ve already seen how compact and simple [RL controllers](https://www.deepmind.com/blog/accelerating-fusion-science-through-learned-plasma-control) can be while emulating the same behavior as control-theory controllers with more generality and in situations that typically warranted strong guarantees of safety. However, I’m not married to RL or AI in general - I simply think that it offers a good solution to this problem.

Taleb in his book Antifragile talks about stressors to systems and I loosely interpret them as any form of disruption to the system outside of the anticipated use cases. For example, a large traffic spike is a stressor that risks crashing many services or severely reducing customer retention.

We need to combine long-term or cyclical stressors with short-term stressors and input them into something I call the systems layer. The systems layer understands the system intimately, not just as a collection of services but the holistic system as a whole. It can infer system dynamics to figure out what sequence of actions caused issues and bottlenecks and what sequence of actions solved it. It understands how to apply hypotheses to the dependency graph to understand how the system behaves given some hypothetical situation. 

What if we could intelligently query the system so that:

1. primitives can actively tune themselves based on immediate stressors and historical data that’s been filtered so only relevant information comes up
2. we can ask where the system failed before and recreate those scenarios to test primitives, i.e. hypothesis testing

## System Primitives

My first thought was to create a learning agent that can adjust parameters on a config file since systems are strung together with config files as interfaces - e.g. the agent could tune the heap size before the garbage collector started according to some traffic distribution. A config file has very specific assumptions about the system/component, the types and the format. However, it’s not an intrinsic part of the system, it’s just an interface to the system. An agent that adjusts config files would be easier to sell and integrate into tech stacks, but we need to make assumptions about the behavior of the underlying service and integrity of the interface (e.g. fields in the config that get ignored in the code). This suggests to me that config files are really second class system primitives. So let’s introduce a first class system primitive by highlighting the difference between a first class and second class primitive: a first class primitive is an intrinsic aspect of the system and can evolve with the system whereas a second class primitive is a proxy to the system and requires collaboration between teams. Moreover, it can involve shims and careful engineering work to migrate to new deployments.

Is this abstraction useful for businesses? Or is the distinction between first and second class primitives just pedantic philosophizing? I don't think so. Instead, I ask you: why must we limit ourselves to such rigid proxied system primitives? Instead, what if we had tunable and intrinsic system primitives that could adapt and learn from new stressors? Why can’t we build systems that intrinsically exploit stressors to adapt to them?

For a business serving customers, stability and correctness translate directly to customer retention. However, these needs have changed and even startups can start to ingest thousands of GBs of data every day, serve thousands of customers within a relatively short period of time, all while battling cloud costs in the name of growth. On top of that, when systems inevitably break because some of config issues, unexpected traffic distributions, latent system bottlenecks, etc. we tend to assign multiple engineers to fix those issues, costing the business money and, what is infinitely more valuable, time. Even worse, we tend to see debugging as a function of intelligence rather than a function of operational excellence. I want to sum up my sentiment with this analogy: _chaos testing is to systems as what sanitizers are to C++ in that both are afterthoughts_. If our needs and capabilities have changed, why haven’t the systems underpinning the modern internet changed?

## System/AI Primitives

Generally, a lot of system-level problems can be framed as some optimization problem and I’m going to explore what these first class primitives could look like by introducing a few examples that solve assignment, scheduling and search problems. Note that these are just a subset of what we can build and the core idea spans across use cases.

## Assignment Problems

### Load Balancing

The goal of load balancing is to make sure that we distribute requests, jobs and workloads to all available server, workers and buckets in a way that doesn’t overwhelm any one bucket. We can do this with consistent hashing or rendezvous hashing if we expect any of the nodes to experience downtime but the basic idea is to facilitate some sort of an assignment problem. These aren’t cryptographic hash functions so we can use a hash function that’s parallelizable and pipeline-able, so long as we can bound entropy on each bucket and on the collective. However, what if I ask you to implement anti-affinity, affinity, sparsity and capacity (e.g. heterogeneous compute)? We want to embed all that information and other biases into the hash function to direct network flow efficiently. Let’s go one step further: what if we can use empirical data to tune all the load balancers based on short- and long-term trends so that we can redirect traffic across the entire network topology. Even large companies like Ali Baba (I can’t find the article but trust me (TM). They describe running into hot spotting issues during Christmas despite designing a [resilient system](https://note.com/kenwagatsuma/n/n33f1caaab7cd)) have had load balancer issues during Black Swan events.

There are ideas from [image retrieval](https://arxiv.org/pdf/2101.11282.pdf) or general [kernel-based hashing](https://wiki.math.uwaterloo.ca/statwiki/index.php?title=kernelized_Locality-Sensitive_Hashing) on creating kernels that can embed biases into hash functions. Kernels embed information like anti-affinity, affinity, sparsity and capacity and effectively parameterize hash functions. So training a kernel corresponds to a hash function biased on the number of gpus, cpus, sticky requests, etc. For this specific load balancing problem, instead of minimizing entropy during training, we can maximize entropy predicated on anti-affinity, capacity, sparseness, etc. - e.g. images/requests that are more unlike each other are placed in the same bucket. These hash functions can be used with consistent or rendezvous hashing without any more integration.

The real benefits of the kernel-based hash only become clearer with a holistic view of the system - you can redirect requests based on how downstream services are shaping traffic, how backed up downstream queues are or which versions are being deployed. With an AI agent tuning this kernel and consequently redirecting requests or jobs based on dynamic shifts in traffic and system health across the organization, organizations can realize optimizations that were previously inaccessible without large engineering teams and strong engineering cultures.

This differentiable kernel can be interpreted as a first class system primitive. Why is it important that it is differentiable? It's not but gradient-based optimization methods have proven to be scalable and relatively inexpensive.

_Kernels are just one interpretation of first class system primitives._

We can think of this assignment problem more abstractly: it could also be a RL agent that learns how to send requests to the right node depending on system characteristics. And the policy that it learns from interacting with the system can be a system primitive.

Suppose we have a hundred load balancers, each with its own set of parameters, traffic distributions and environments. Suppose we want to onboard a new load balancer but we don’t want to train this hash function/RL agent/primitive again. This is where we can train a meta kernel model that produces kernels given system information like traffic distributions. The great part about this meta model is that it can be reused across organizations, companies and scales.

### Sharding

Like load balancing, sharding is also an assignment problem. However, it differs from load balancing in how the underlying model is trained: re-sharding can be very expensive so we would want to bound the deviation in the kernel and include the cost of re-sharding in its cost function. Moreover, the frequency at which we change parameters would be drastically reduced compared to that of the load balancer and that cost would be embedded into the objective function.

Sharding relocates certain rows and columns in a database to make it easier to scale read and write workloads. However, with more complex data queries which include joins, database caches, external caches sitting on top of databases, and frequency of changing values and quantities, sharding becomes a high-dimensional optimization problem. High-dimensional optimization problems are exactly what existing AI solutions are great for.

A temporally localized view of incoming traffic might create a tuning that doesn’t address long-term trends. And it doesn’t account for human-centric traditions like Christmas-time load. An empirical model trained with 11 months of data with slight disruptions might never understand why the system becomes very busy on the 12th month. Or it might consider the 12th month as an anomaly.

We can use the same kernel idea as before to figure out how to assign rows to a shard.

## Search Problems

### Query Optimizers

When queries aren't optimized correctly, it can introduce delays across the system because databases underpin a lot of systems. Solutions like [airops](https://www.airops.com/blog/using-ai-to-optimize-your-sql-queries) and [postgres.ai](https://postgres.ai/products/joe) already exist by creating optimized SQL queries but they're not first class primitives. They're interfaces to the query optimizer, which [needs to be tediously maintained, especially as the system’s execution and storage engines evolve](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf). There are benefits to understanding the history of calls to a database to exploit caches and improve index selection. What does a first class query optimizer primitive look like?

[Neo](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf) proposes a neural network based query optimizer that is comparable to optimizers from Microsoft and IBM by essentially picking the 'optimal' query and execution plans. More importantly, it can dynamically adapt to [incoming queries, building upon its successes and learning from its failure](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf).

This neural network fits the model of an system primitive. By adjusting the weights of this neural network, we can address novel situations based on the environment, storage needs, and traffic distribution. Antifragile system primitives can come in various shapes and formats - the invariance is that they can learn.

## Scheduling Problems

### Schedulers

So far, we've talked about creating first class system primitives for assignment and search problems. What about scheduling problems? What about scheduling and assignment problems? When you submit jobs to your cluster, you request system resources like compute and memory. But the system doesn't precisely know how much compute and memory you'll need in the duration of your program execution. What happens if there's a large burst of ML training jobs and you need to quickly provision compute and memory for those jobs? You could over-provision and waste resources or under-provision and redo or slow down the task in a e.g. map reduce task. There are a few [examples](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview) that have already used AI to find a better packing solution.

We also see scheduling problems in process schedulers for [hyper-V](https://www.vmware.com/content/dam/digitalmarketing/vmware/en/pdf/techpaper/vmware-vsphere-cpu-sched-performance-white-paper.pdf). We've already seen RL agents handle generic scheduling and assignment problems and do considerably [better](https://towardsdatascience.com/reinforcement-learning-for-production-scheduling-809db6923419) than conventional optimization techniques, suggesting that we could realize significant gains in many aspects of the system given the ubiquity of scheduling+assignment problems.

Using a RL agent to schedule+assign also fits the criteria for a first class system primitive by having the policy mirror the internal system state.

## Hierarchal Autonomous Agents

If every system component has a first class primitive that learns how to tune its parameters based on external information, we are still only realizing local efficiencies. However, we could go one step further and realize global efficiencies by sharing information between system primitives or having a hierarchical agent that can negotiate or direct individual system primitives based on larger organizational objectives like cost, reliability or throughput. This looks like a [multi-agent](https://arxiv.org/pdf/1911.10635.pdf) problem where the agents must cooperate to realize their common goals.

I’d like to motivate the need for a central controller with an example of when a new endpoint is added. Suppose the new endpoint is serving a small group of customers initially who are read-heavy and they see terrible performance. Maybe we should re-shard so that we can handle the new number of read requests (unlikely first course of action but bear with me)? Or maybe instead we should create a cache in front of the database? Or maybe the current system can handle the traffic but the system can exploit database caches if it ensures that the read requests from the new endpoint are issued from the same nodes that the write requests are assigned to? How do we decide which of these solutions is globally better? What objectives are driving decisions at organizations: money saved? strategic positioning? vendor lock-in? reducing complexity?

Eventually, it’d be great to be able to train joint text-policy embeddings so we can ask the central controller, in natural language, to optimize on a set of objectives - yet another use case that LLMs have made possible.

## Deployments

How do we deploy a system like this? How can we as humans interpret outcomes? How can we debug these solutions? It's not as straightforward to debug a declarative system. But there are more established methods to productionize AI models like [Claypot](https://www.claypot.ai/) that at least create precedence for how to design these systems, ensure that we can detect model drift and rollback when the model quality starts slipping.

There are two deployment methods:

1. We re-train a model or call the meta kernel model every time we detect model drift
2. We use an online-learning, [continual learning model](https://arxiv.org/abs/2306.13812v2) that can inherently adapt to new situations

While the second solution is more elegant, the first solution is currently more established and easier to reason about.

## Conclusions

When I search for AI in distributed systems or AI in large scale systems, I see [articles](https://engineering.fb.com/2021/07/15/open-source/fsdp/), papers and blogs on how to create systems that can train or do inference - i.e. create systems that can help create models. But I haven't seen many instances on how to use AI to boost the performance of end-to-end systems in production. [Jane Street](https://blog.janestreet.com/what-the-interns-have-wrought-2020/), Databricks and [Google](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview) have invested in using AI in their infrastructure but I largely see this idea discussed in academia suggesting to me that this idea has enough merit to investigate and potentially enough that companies are willing to put their weight behind it. Note, I'm not talking about AIOps or augmentation of systems, e.g. forecasting of traffic to provide signals to an autoscaler but rather I'm talking about something more in the vein of [Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35). I'm not sold on an empirical and declarative way of writing software. However, I am convinced that framing optimization problems as declarative problems, as I believe many systems problems are, have benefits. Especially if it's adapting to new situations on its own.

Software 2.0 focuses on the individual services while I'm proposing a different approach for building systems by focusing on system aspects like antifragility.

Antifragile systems present a new way to think about software systems in general. I wouldn't be surprised, if in 10 years, this becomes the de facto paradigm of creating systems that can realize both holistic and local efficiencies, reduce the SRE burden considerably and most importantly, reduce system complexity.

---

## Appendix

### Autoscalers

There are autoscalers that use linear or near-linear models as opposed to very non-linear models and the effect is felt most strongly with [thundering herd issues](https://engineering.fb.com/2015/12/03/ios/under-the-hood-broadcasting-live-video-to-millions/). The problem with using a linear or near-linear model is that for large bursts in traffic, the number of pods doesn't increase fast enough and when it finally does, it's too late since the customers have already experienced significant lag. This happens because if the controller is overdamped, the settling time is too high. And if it's underdamped, we have ringing issues. So we need to tune the controller just right for it to address these traffic bursts at a fast enough rate. But the traffic distribution is constantly changing so we need more complex systems to forecast that and change the tuning of the controller. My point is that, in production, these systems can get complex and at some point, I would argue for a simpler and more compact RL model to handle the autoscaling task.

Something else I've been trying to argue for is to propagate system signals, e.g. the fact that an autoscaler changed the number of pods to X, across the call-path so that downstream services can react appropriately. So even if system changes across a call-path are only mildly correlated, we can chain system changes before services are impacted and we see spillover and threshold breaches.
