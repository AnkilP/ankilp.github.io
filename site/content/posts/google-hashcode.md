---
title: "Google Hashcode"
date: 2021-03-01T00:14:44-05:00
slug: "hashcode"
description: "Description of my team's solution"
keywords: ["heuristic", "scheduling algorithms", "round-robin"]
draft: false
tags: ["heuristic", "scheduling algorithms", "round-robin", "job shop", "NP-hard"]
math: true
toc: false
---

Update: We participated this year (2022) as well (Top 3% globally - Team name: 6 am vibes). I've started to realize that these problems are very similar... they're some combination of assignment + scheduling problems. And the solutions generally follow the simulated annealing approach.  I hope to compete next year and I'll certainly aim higher to test whether breaking into upper echelons requires a completely different mode of thinking. 

## Introduction

I competed in the online qualification round for Google Hashcode this week (2021) (Team: Not Last Place). We didn't do as terribly as we did last year - we placed 29th in Canada and in the top 30% globally. I wanted to share our solution in this doc and later go over areas of improvement. I welcome any and all comments and questions - send me an email <ankil335@gmail.com>.

## Problem

We have a city with intersections and each of these intersections have a street light that control which cars can go through. Each street light has a round-robin schedule - naturally, only one in-degree street can be green at any given time. We repeat the street lights' schedule until the simulation is over. A number of cars want to traverse the city and have pre-defined paths. We want to create a schedule for the street lights such that we minimize the amount of time it takes each car to get to its destination (any car that doesn't reach its destination has a score of 0). All streets are connected by intersections and vice versa. It is my understanding that there are no dead ends. Moreover, only one car can pass through the intersection every second. There is no limit to the number of cars that can be queued at the intersection.

### Input

The input essentially includes information for each street and each car that drives in the city. Each car has a pre-defined route of streets that the car traverses. A car takes $L_{street_A}$ time units to cross $street_A$. Information for each street includes the intersections that connect the beginning and the end of the street. 

### Output

For each intersection, we expect to see the schedule for each street light. This schedule is repeated until the simulation time is over. The score is determined by the number of cars that reach their destination plus the time difference between when a car reaches its destination and when the simulation ends.

## Approach

We needed to grapple with a few questions:
* Does optimizing the schedule for each individual street light imply optimizing the schedule for the city as a whole?

* With the limited time available, is it worth pursuing an optimal solution or is it better to use a heuristic? How good can this heuristic get (and still complete the problem in time)?

For the first question, we initially assumed it was true but when we used more interesting datasets, we realized that this was not an assumption we could always make (if ever). The problem was also similar to problems I had seen in my Intro to Optimization class with the job shop problem - i.e. there are a number of jobs with tasks that need to be done in order and a number of machines that can process these jobs. The difference was that we needed to figure out how long to run each machine (street light) for each job. The objective function remains the same: minimize the makespan. The job shop problem is a NP-hard problem and variations of the job shop problem that are closer to this specific problem are NP-complete or NP-hard [1]. As a result, we decided to use a heuristic.

For the second question, we thought about what kind of heuristic we'd use. Given the nature of the round-robin schedule of the street-lights, we would need to determine how long to keep the green light on for each street. This amount can't be dynamic - once it's set, it has to be used for the entire simulation. 

The baseline we came up with was an average of the cars that visit the intersection on their route. In most cases, the cars won't have to wait. If there's a large burst of cars (e.g. 2x the average), then the second half of the cars will wait considerably longer than the first half since they need to wait for the schedule to repeat.If there are no cars queued, then we're wasting time that could otherwise be given to another street. This simple baseline net us 8 million points. 

There are a few problems already - we calculated the average based on iteration number not time. In other words, if $car_A$ has a route: $X,Y,Z$ and $car_B$ has a route: $Q,W,Z,V,X$ with different $L$'s for each unique street, then we would calculate the average for Z as 2/5 since Z appears twice in the third iteration over 5 iterations. If, however, streets $X$ and $Y$ only take 1 time unit to traverse and $Q$ takes 100 time units to traverse, then it's hard to make the case that the green light for $Z$ should be turned on for more than 1 time unit since at any given time, at most one car will have stopped at $Z$. By incorporating $L$ values into our baseline, we roughly got another 1 million points. The only thing left was to predict the wait times.

Interestingly, by this point, we still weren't getting full points on the example test case. The example test case had two cars, both of which would go to the same intersection, but at different times and from different streets. Our algorithm was designed for maximizing throughput, not minimizing wait times and assigned that street light a schedule of 1 time unit:1 time unit ($street_A$:$street_B$) since that would be the capacity. We didn't account for the initial positions of the cars. In the example test case, the second car (coming from $street_B$) gets to the intersection after 2 time units, but because of the schedule, has to wait 1 time unit before it can be moved through. And by then, it's too late. The ideal schedule would be 2:1. That way, the second car would have a green light as soon as it got to the street light. Unfortunately, at this point, we didn't have enough time to flesh out a general solution to this problem. We discussed several solutions:

* propagate slack times across the intersections
* modify job shop Mixed Integer Linear Programs (MILP) to address this problem

Moving forward, we should look at the distribution of the dataset. It could give us some information on what kind of heuristics to pursue (e.g. how bursty the cars are) and whether other algorithms could help. In the past, ILP and LP implementations have been too slow for large datasets, but investing time in learning how to use Google's OR-tools could be useful. Genetic algorithms were floated around as a way to determine the optimal schedule for each street light.


## Misconceptions

I had seen similar problems in my OS class with single threaded CPU's. The scheduling algorithms used are along the lines of Shortest Remaining Time First, First-Come-First-Serve, etc. However, I can only make those optimality guarantees if I can split up the time given to a street within a single schedule.

References: <http://people.idsia.ch/~monaldo/papers/EJOR-varJsp-05.pdf>.

