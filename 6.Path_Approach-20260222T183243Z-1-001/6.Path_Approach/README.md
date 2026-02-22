# Path Approach


* Estimate the friend's influence based on the intention model. We argue that a negative mood is transferable from one person to another during social interactions.
  Therefore, the negative status updates and the intention behind them may spread through friendship networks resulting in a higher occurrence of similarly negative status updates. Since the proposed intention model represents the true motivation of individuals, the more similar the intention models of two nearby users are, the more likely one user gets affected by the other one.

* To build a friendship network with users whose assessment results are available, we run the shortest path for the whole network and find all the connections with available assessment results to the target user. Similarly, the influence score of user U's depressed neighbors is calculated as:

  ![image](https://github.com/jeelan-ds786/Detecting-User-Level-Depression-Using-Social-Network-Text-Analysis-/assets/97782415/b005e3b3-f7d8-4f29-9fb5-a5accc890f54)

* Calculating Influence Score :

  ![image](https://github.com/jeelan-ds786/Detecting-User-Level-Depression-Using-Social-Network-Text-Analysis-/assets/97782415/b3b6ebfc-dd01-4e1c-894d-b32ef2dd453d)

   * InfScoreINT, we create two extra features to store the influence score values for each user in our data set using both approaches. The friends’ influence features indicate how
much users are influenced by their depressed friends. The values are between 0 to 1.


<hr>
<br>


## Path Approach flow chart :






  ![image](https://github.com/jeelan-ds786/Detecting-User-Level-Depression-Using-Social-Network-Text-Analysis-/assets/97782415/78755c09-142e-42c0-bde9-3b3d162df28a)

