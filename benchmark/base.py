from abc import ABC, abstractmethod
import numpy as np

class Benchmark(ABC):
    """
    Abstract Class
    """


    def __init__(
        self, 
        path,
        shape
    ):
        """
        shape: shape of the input excluding batch_size,
        e.g.  images: [resolution, resolution, channels], graphs: [num_nodes, feature_size, num_edges]
        """
        self.shape = shape
        self.load_model(path)

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def get_dummy(self, shape):
        """
        Get a dummy variable of a given shape
        """
        pass
    
    @abstractmethod
    def time_model(self, model, dummy_input):
        """
        Get the time spent for the inference, measured in ms
        """
        pass

    def warm_up(self, model, batch_size, repetitions):
        dummy_input = self.get_dummy([batch_size]+self.shape) 
        for _ in range(repetitions):
            _ = self.time_model(model, dummy_input)


    def get_latency(self, model, batch_size, repetitions):
        timings=np.zeros((repetitions,1))

        # MEASURE PERFORMANCE
        for rep in range(repetitions):
            dummy_input = self.get_dummy([batch_size]+self.shape)
            timings[rep] = self.time_model(model, dummy_input)

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)    
        return mean_syn, std_syn
    

    def get_throughput(self, model, batch_size, repetitions):

        total_time  = 0
        for _ in range(repetitions):
            dummy_input = self.get_dummy([batch_size]+self.shape)
            total_time += self.time_model(model, dummy_input)/1000 #to convert in second (original in ms)

        throughput =   (repetitions*batch_size)/total_time  # n_images/total_time 
        return(throughput)



    #def get_optimal_resolution(self, model):
    #    self.warm_up(model)
    #    optimal_resolution = 128
    #    for resolution in [256, 512, 1024, 2048, 4096]:
    #        dummy_input = self.get_dummy((1, resolution, resolution, self.channels))
    #        try:
    #            _ = self.time_model(model, dummy_input)
    #            optimal_resolution = resolution
    #        except RuntimeError as e:
    #            print(e)
    #            break
    #    return(optimal_resolution)

    #def get_optimal_batch_size(self, model):
    #    self.warm_up(model)
    #
    #    optimal_batch_size = 1
    #    for batch_size in [32, 64, 128, 256, 512, 1024]:
    #        dummy_input = self.get_dummy((batch_size, self.resolution, self.resolution, self.channels))
    #        try:
    #            _ = model(dummy_input)
    #            optimal_batch_size = batch_size
    #        except RuntimeError as e:
    #            print(e)
    #            break
    #    return(optimal_batch_size)



    def metrics(
        self,
        latency_batch_size,
        throughput_batch_size,

        warm_up_repetitions = 50,
        latency_repetitions = 300,
        throughput_repetitions = 100,
        repetitions = 2
    ):
        if(latency_batch_size is not None):
            for _ in range(repetitions):
                self.warm_up(self.model, latency_batch_size, warm_up_repetitions)
                mean, std = self.get_latency(self.model, latency_batch_size, latency_repetitions)

            print('Latency, average time (ms):', mean)
            print('Latency, std time (ms):', std)

        if(throughput_batch_size is not None):
            #optimal_batch_size = get_optimal_batch_size(model)
            for _ in range(repetitions):
                self.warm_up(self.model, throughput_batch_size, warm_up_repetitions)
                throughput = self.get_throughput(self.model, throughput_batch_size, throughput_repetitions)

            print('Final Throughput (obj/s):',throughput)
