from util import *

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, name, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
        self.name = name
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        # followings are for directed

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.015
        
        self.momentum = 0.7

        self.print_period = 6000
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 6000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return

        
    def cd1(self, visible_trainset, n_iterations=int(60000/10*20)):
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
        visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
        n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        print("learning CD1")
        recon_loss = []
        plotting = False
        n_samples = visible_trainset.shape[0]
        # print(visible_trainset.shape)
        loss_list = []
        results_list = []
        error = []  # Storing error per iteration
        elements = int(n_samples / self.batch_size)
        current_epoch = 1  # Initialize current epoch as the first one
        it = 0
        while True:
            if it >= n_iterations:
                # 60000/batch_size iterations = 1 epoch
                break
            
            for epoch in range(elements):
                # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1. you may need to
                #  use the inference functions 'get_h_given_v' and 'get_v_given_h'. note that inference methods returns
                #  both probabilities and activations (samples from probablities) and you may have to decide when to use
                #  what.

                index_init = epoch
                # index_init = int(it % elements)
                index_stop = int((index_init + 1) * self.batch_size)
                index_init *= self.batch_size
                # print(index_init, index_stop)
                v_0 = visible_trainset[index_init:index_stop, :]
                ph_0, h_0 = self.get_h_given_v(v_0)
                pv_1, v_1 = self.get_v_given_h(h_0)
                # if self.is_top:
                #     v_1[:, -self.n_labels:] = v_0[:, -self.n_labels:] # clamp labels
                ph_1, h_1 = self.get_h_given_v(v_1)

                # [TODO TASK 4.1] update the parameters using function 'update_params'

                # self.update_params(v_0, h_0, v_1, h_1)
                self.update_params(v_0, ph_0, v_1, ph_1)
                it += 1
                if plotting:

                    if current_epoch >= 10:
                        hidden_restored = self.get_h_given_v(v_0)[1]
                        restored_image = self.get_v_given_h(hidden_restored)[1]
                        loss_function = np.linalg.norm(v_0 - restored_image) / self.batch_size
                        loss_list.append(loss_function) 
                        if it == elements - 1:
                            results_list.append(np.array(loss_list).sum() / len(loss_list))  # Append avg loss epoch
                            loss_list = []

                    if it % self.print_period == 0:
                         loss_function = np.linalg.norm(v_0 - v_1) / self.batch_size
                         print("\niteration=%7d recon_loss=%4.4f" % (it, loss_function))


                    if it % self.batch_size == 0:
                        hidden_restored = self.get_h_given_v(v_0)[1]
                        restored_image = self.get_v_given_h(hidden_restored)[1]
                        error.append(np.linalg.norm(v_0 - restored_image))

                  
                if it % self.rf["period"] == 0 and self.is_bottom:
                    
                    viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])

                # print progress
                
                if it % self.print_period == 0:
                    # Reconstruction Error (L2 norm)
                    # k = 1 (only repeat 1 step)
                    _, h_k = self.get_h_given_v(visible_trainset)
                    _, v_k = self.get_v_given_h(h_k)
                    t = np.linalg.norm(visible_trainset - v_k)
                    print ("iteration=%7d recon_loss=%4.4f"%(it, t))
                    recon_loss.append(t)
                    # Sample from trainset to see the reconstructed pictures 
                    # see comparisons in RECON.xxxxxx.png
                    if self.is_bottom:
                        n_S = 25
                        samples = visible_trainset[:n_S, :].T
                        samples_recon = v_k[:n_S, :].T
                        samples = np.hstack((samples, samples_recon))
                        samples = samples.reshape((self.image_size[0],self.image_size[1],-1))
                        viz_rf(samples, it, [5, 10], tp='RECON')
                
            
            current_epoch += 1

        if plotting:
            plt.plot(range(len(error)), error)
            plt.xlabel("Batch Number")
            plt.ylabel("Error/Loss")
            plt.show()

        plt.plot(recon_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Reconstruction loss (L2 norm)')
        plt.savefig(f"{self.name}.reconstruction_loss.png")
        plt.close('all')
        # return results_list

    

    def update_params(self, v_0, h_0, v_k, h_k):
        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
        v_0: activities or probabilities of visible layer (data to the rbm)
        h_0: activities or probabilities of hidden layer
        v_k: activities or probabilities of visible layer
        h_k: activities or probabilities of hidden layer
        all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters

        # h_0 and h_k here should be the probability
        # use momentum updating
        self.delta_weight_vh = self.momentum * self.delta_weight_vh + (1 - self.momentum) * ((v_0.T @ h_0) - (v_k.T @ h_k)) / self.batch_size
        self.delta_bias_v = self.momentum * self.delta_bias_v + (1 - self.momentum) *  np.mean((v_0 - v_k), axis=0)
        self.delta_bias_h = self.momentum * self.delta_bias_h + (1 - self.momentum) *  np.mean((h_0 - h_k), axis=0)

        self.weight_vh += self.learning_rate * self.delta_weight_vh
        self.bias_v += self.learning_rate * self.delta_bias_v
        self.bias_h += self.learning_rate * self.delta_bias_h

        # return


    def get_h_given_v(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args:
        visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
        tuple (p(h|v), h)
        both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None

        # n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below)
        prob_h_given_v = visible_minibatch @ self.weight_vh + self.bias_h
        prob_h_given_v = sigmoid(prob_h_given_v)
        h_given_v = sample_binary(prob_h_given_v)

        return prob_h_given_v, h_given_v



    def get_v_given_h(self, hidden_minibatch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
        hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
        tuple (p(v|h), v)
        both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        # n_samples = hidden_minibatch.shape[0]

        # Calculate total input to the visible layer
        support = hidden_minibatch @ self.weight_vh.T + self.bias_v

        # Create arrays
        prob_v_given_h = np.zeros_like(support)
        v_given_h = np.zeros_like(support)

        if self.is_top:
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            prob_v_given_h[:, :-self.n_labels] = sigmoid(support[:, :-self.n_labels])
            prob_v_given_h[:, -self.n_labels:] = softmax(support[:, -self.n_labels:])

            # Sample activities
            v_given_h[:, :-self.n_labels] = sample_binary(prob_v_given_h[:, :-self.n_labels])
            v_given_h[:, -self.n_labels:] = sample_categorical(prob_v_given_h[:, -self.n_labels:])
        else:
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)           
            prob_v_given_h = sigmoid(support)
            v_given_h = sample_binary(prob_v_given_h)

        return prob_v_given_h, v_given_h




    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        # self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        # assert self.weight_v_to_h is not None

        # n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 
        support = visible_minibatch @ self.weight_v_to_h + self.bias_h
        prob_h_given_v = sigmoid(support)
        h_given_v = sample_binary(prob_h_given_v)

        return prob_h_given_v, h_given_v


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        # assert self.weight_h_to_v is not None
        
        # n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            
            raise Exception('This RBM is at the top, it has no directed connections!!')
            
        else:
                        
            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             
            support = hidden_minibatch @ self.weight_h_to_v + self.bias_v
            prob_v_given_h = sigmoid(support)
            v_given_h = sample_binary(prob_v_given_h)
            
        return prob_v_given_h, v_given_h
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        
        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    
