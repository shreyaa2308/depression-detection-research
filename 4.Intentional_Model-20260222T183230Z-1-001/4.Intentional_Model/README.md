# Intentional Model


  * In our dataset, most of the annotated content belongs to three main acts: assertive, directive, and expressive. Therefore, we focus on predicting these three acts as they are the most common act types with respect to depressed users in the dataset. We randomly split our dataset by allocating 80% for our training set and 20% for our hold-out test set. On the training set, we used 10-fold cross-validation for all methods. We compare the performance of the four classifiers with a baseline classifier. The baseline classifier is designed using the existing text classification techniques.
  * We generated a Sample dataset to train the model, created 50 statements in each act and trained Machine Learning Model

<hr>

# The Three Main Acts :

## Assertive :

  * Assertive speech acts are used to state facts or opinions. They are typically used to inform, report, or describe something.

        Examples: "I think the weather is going to be nice today."
        Functions: To state a fact, to express an opinion, to report on something, to describe something.


## Directive :

  * Directive speech acts are used to get someone to do something. They are typically used to order, request, or suggest something.

        Examples: "Please pass the salt."
        Functions: To order someone to do something, to request that someone do something, to suggest that someone do something.


## Expressive :

  * Expressive speech acts are used to express emotions or feelings. They are typically used to apologize, thank, congratulate, or express sympathy.

        Examples: "I'm sorry I'm late."
        Functions: To apologize, to thank someone, to congratulate someone, to express sympathy.




![image](https://github.com/jeelan-ds786/Detecting-User-Level-Depression-Using-Social-Network-Text-Analysis-/assets/97782415/6c0a41ec-f553-47b6-a085-94faac329ddc)



<br>




## More example statements on speech of acts:



* `Assertive`

  * "The sky is blue."
  * "I believe that climate change is real."
  * "I'm not feeling well today."




* `Directive`

  * "Close the door, please."
  *  "Can you help me with this?"
  * "Let's go for a walk."



* `Expressive`

  * "I'm so happy to see you!"
  * "I'm sorry for your loss."
  * "Thank you for the gift."
