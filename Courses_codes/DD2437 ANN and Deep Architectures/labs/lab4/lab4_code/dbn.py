from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(name= 'vis--hid',ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(name = 'hid--pen', ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(name='pen+lbl--top',ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 50
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        
        vis = true_img # visible layer gets the image data
        
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
        
        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.
        vis_hid = self.rbm_stack['vis--hid']
        hid_pen = self.rbm_stack['hid--pen']
        pen_lbl_top = self.rbm_stack['pen+lbl--top']

        _, h_1 = vis_hid.get_h_given_v_dir(vis)
        #print(h_1.shape)
        _, h_2 = hid_pen.get_h_given_v_dir(h_1)
        h_2_lbl = np.hstack((h_2, lbl))
        diff_list = []
        for _ in range(self.n_gibbs_recog):
            # print(_)
            _, top = pen_lbl_top.get_h_given_v(h_2_lbl)
            _, h_2_lbl = pen_lbl_top.get_v_given_h(top)
            
            diff = 1 - np.mean(np.argmax(h_2_lbl[:, -self.sizes["lbl"]:],axis=1)==np.argmax(true_lbl,axis=1))
            diff_list.append(diff)


        predicted_lbl = h_2_lbl[:, -self.sizes["lbl"]:]
            
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        plt.plot(diff_list)
        plt.xlabel('Gibbs samples')
        plt.ylabel('Misclassification rate')
        plt.savefig("testing_label_convergence.png")
        plt.close('all')
        return

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl
        # print(lbl)
        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).
        vis_hid = self.rbm_stack['vis--hid']
        hid_pen = self.rbm_stack['hid--pen']
        pen_lbl_top = self.rbm_stack['pen+lbl--top']

        ramdom_img = sample_binary(0.5 * np.ones((n_sample, self.sizes["vis"])))
        # records.append( [ ax.imshow(ramdom_img.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
        # print(ramdom_img)
        _, h_1 = vis_hid.get_h_given_v_dir(ramdom_img)
        _, h_2 = hid_pen.get_h_given_v_dir(h_1)
        h_2 = np.hstack((h_2, lbl))
        last_h2 = h_2
        for i in range(self.n_gibbs_gener):

            h_2[:, -self.sizes["lbl"]:] = lbl # clamp labels
            # print(h_2[:, -self.sizes["lbl"]:])
            _, top = pen_lbl_top.get_h_given_v(h_2)
            _, h_2 = pen_lbl_top.get_v_given_h(top)
            # print(np.sum(np.abs(h_2 - last_h2)))
            last_h2 = h_2
            _, h_1 = hid_pen.get_v_given_h_dir(h_2[:, :-self.sizes["lbl"]])
            _, vis = vis_hid.get_v_given_h_dir(h_1)
            # vis = np.random.rand(n_sample,self.sizes["vis"])
            records.append( [ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
        # import matplotlib.animation as animation
        # writervideo = animation.FFMpegWriter(fps=60)
        anim = stitch_video(fig,records).save("./image_generation/%s.generate%d.gif"%(name,np.argmax(true_lbl)), )  
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries 
        to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        # vis_hid = self.rbm_stack['vis--hid']
        # hid_pen = self.rbm_stack['hid--pen']
        # pen_lbl_top = self.rbm_stack['pen+lbl--top']

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
        except IOError :

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily
        
            print ("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """            
            self.rbm_stack['vis--hid'].cd1(vis_trainset, n_iterations)
            
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights() 

        try:
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()

        except IOError:
            print ("training hid--pen")
            """ 
            CD-1 training for hid--pen 
            """ 
            _, h_1 = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis_trainset)
            self.rbm_stack['hid--pen'].cd1(h_1, n_iterations)                
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")     
            self.rbm_stack["hid--pen"].untwine_weights()

        try:
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")  

        except IOError:
            print ("training pen+lbl--top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            _, h_1 = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis_trainset)
            _, h_2 = self.rbm_stack['hid--pen'].get_h_given_v_dir(h_1)   
            h_2 = np.hstack((h_2, lbl_trainset))
            self.rbm_stack['pen+lbl--top'].cd1(h_2, n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):            
                                                
                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.

                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.

                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.
                
                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
