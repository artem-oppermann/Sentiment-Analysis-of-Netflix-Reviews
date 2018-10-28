# Sentiment-Analysis-of-Netflix-Reviews
Sentiment Analysis with bi-LSTM's recurrent neural network's.


The model consists of a bidirectional one layer long short term memory reccurent neural network. It is trained on roughly 5000 one-sentence labeled positiv and negativ Netflix reviews. The model's purpose is to learn to recognize a sentiment of a review and classify it whether as a positiv or a negativ review. 

The training of the model needs only about three epoch to reach an accuracy of roughly 75% on the validation set consisting of 1000 reviews. After that the accuracy goes never beyond 78 %. It can be observed that the training accuracy goes very fast towards 90-95% due to overfitting of the model. To prevent this I use a L2 regularization.
Here is an example for the training for 5 epochs and its summarys: 

    epoch_nr: 0, batch_nr: 200/603, train_loss: 0.749, train_acc: 0.556, val_acc: 0.596 
    epoch_nr: 0, batch_nr: 300/603, train_loss: 0.740, train_acc: 0.596, val_acc: 0.544 
    epoch_nr: 0, batch_nr: 400/603, train_loss: 0.733, train_acc: 0.570, val_acc: 0.662 
    epoch_nr: 0, batch_nr: 500/603, train_loss: 0.723, train_acc: 0.657, val_acc: 0.670 
    epoch_nr: 0, batch_nr: 600/603, train_loss: 0.701, train_acc: 0.667, val_acc: 0.664 
    epoch_nr: 1, batch_nr: 100/603, train_loss: 0.633, train_acc: 0.743, val_acc: 0.674 
    epoch_nr: 1, batch_nr: 200/603, train_loss: 0.618, train_acc: 0.734, val_acc: 0.708 
    epoch_nr: 1, batch_nr: 300/603, train_loss: 0.580, train_acc: 0.769, val_acc: 0.720 
    epoch_nr: 1, batch_nr: 400/603, train_loss: 0.573, train_acc: 0.757, val_acc: 0.726 
    epoch_nr: 1, batch_nr: 500/603, train_loss: 0.553, train_acc: 0.783, val_acc: 0.736 
    epoch_nr: 1, batch_nr: 600/603, train_loss: 0.542, train_acc: 0.785, val_acc: 0.736 
    epoch_nr: 2, batch_nr: 100/603, train_loss: 0.452, train_acc: 0.849, val_acc: 0.732 
    epoch_nr: 2, batch_nr: 200/603, train_loss: 0.412, train_acc: 0.863, val_acc: 0.748 
    epoch_nr: 2, batch_nr: 300/603, train_loss: 0.422, train_acc: 0.841, val_acc: 0.754 
    epoch_nr: 2, batch_nr: 400/603, train_loss: 0.419, train_acc: 0.847, val_acc: 0.764 
    epoch_nr: 2, batch_nr: 500/603, train_loss: 0.416, train_acc: 0.853, val_acc: 0.764 
    epoch_nr: 2, batch_nr: 600/603, train_loss: 0.435, train_acc: 0.846, val_acc: 0.768 
    epoch_nr: 3, batch_nr: 100/603, train_loss: 0.322, train_acc: 0.908, val_acc: 0.766 
    epoch_nr: 3, batch_nr: 200/603, train_loss: 0.329, train_acc: 0.903, val_acc: 0.774 
    epoch_nr: 3, batch_nr: 300/603, train_loss: 0.316, train_acc: 0.901, val_acc: 0.772 
    epoch_nr: 3, batch_nr: 400/603, train_loss: 0.311, train_acc: 0.896, val_acc: 0.772 
    epoch_nr: 3, batch_nr: 500/603, train_loss: 0.325, train_acc: 0.896, val_acc: 0.776 
    epoch_nr: 3, batch_nr: 600/603, train_loss: 0.338, train_acc: 0.884, val_acc: 0.774 
    epoch_nr: 4, batch_nr: 100/603, train_loss: 0.250, train_acc: 0.946, val_acc: 0.758 
    epoch_nr: 4, batch_nr: 200/603, train_loss: 0.233, train_acc: 0.931, val_acc: 0.778 
    epoch_nr: 4, batch_nr: 300/603, train_loss: 0.251, train_acc: 0.929, val_acc: 0.770 
    epoch_nr: 4, batch_nr: 400/603, train_loss: 0.233, train_acc: 0.929, val_acc: 0.752 
    epoch_nr: 4, batch_nr: 500/603, train_loss: 0.254, train_acc: 0.924, val_acc: 0.756 
    epoch_nr: 4, batch_nr: 600/603, train_loss: 0.270, train_acc: 0.917, val_acc: 0.764 


After the training some samples from the validation set are selected and the model predicts it's sentiment:

    TEST SAMPLES: 

    REVIEW: my big fat greek wedding is not only the best date movie of the year it s also a dare i say it 
            twice delightfully charming and totally american i might add slice of comedic bliss              
    SENTIMENT: positiv 

    REVIEW: what an embarrassment                                               
    SENTIMENT: negativ 

    REVIEW: the central story lacks punch                                             
    SENTIMENT: negativ 

    REVIEW: the latest adam sandler assault and possibly the worst film of the year                                     
    SENTIMENT: negativ 

    REVIEW: quite frankly i can t see why any actor of talent would ever work in a mcculloch production 
            again if they looked at how this movie turned out                      
    SENTIMENT: negativ 

    REVIEW: a wildly entertaining scan of evans career                                           
    SENTIMENT: negativ 

    REVIEW: clooney s debut can be accused of being a bit undisciplined but it has a tremendous offbeat 
             sense of style and humor that suggests he was influenced by some of the filmmakers who have
             directed him especially the coen brothers and steven soderbergh      
    SENTIMENT: positiv 

    REVIEW: michele is a such a brainless flibbertigibbet that it s hard to take her spiritual quest at 
            all seriously                       
    SENTIMENT: positiv 

    REVIEW: the best way to describe it is as a cross between paul thomas anderson s magnolia and david lynch  
            mulholland dr               
    SENTIMENT: negativ 

    REVIEW: really is a pan american movie with moments of genuine insight into the urban heart                                   
    SENTIMENT: positiv 

    REVIEW: the makers of mothman prophecies succeed in producing that most frightening of all movies a mediocre
            horror film too bad to be good and too good to be bad                     
    SENTIMENT: positiv 

    REVIEW: the sundance film festival has become so buzz obsessed that fans and producers descend upon utah 
            each january to ferret out the next great thing tadpole was one of the films so declared this year
            but it s really more of the next pretty good thing    
    SENTIMENT: positiv 

    REVIEW: i don t think this movie loves women at all                                        
    SENTIMENT: negativ 

    REVIEW: even if the ring has a familiar ring it s still unusually crafty and intelligent for hollywood horror   
    SENTIMENT: positiv 

    REVIEW: expands the horizons of boredom to the point of collapse turning into a black hole of dullness 
            from which no interesting concept can escape                          
    SENTIMENT: negativ 

