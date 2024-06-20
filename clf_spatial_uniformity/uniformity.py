# Majority Filter operation
import numpy as np
from typing import Tuple
from scipy.ndimage import generic_filter

def uniformity_index(
        array: np.array, 
        uniformity_threshold: int, 
        kernel_size: int=3, 
        kernel_diagonals: bool=True, 
        target_class: int or str = None, 
        border_mode: str = 'reflect', 
        ignore_nan: bool = True, 
        overall_key: str = 'OVERALL'
) -> Tuple[dict, np.array]:
    """Return 
        (1) a dictionary with the overall and per-class uniformity scores; and,
        (2) the neighbourhood similarity matrix.
    
    Compute the uniformity index of a given classification array.

    Args:
        array (np.array): N-dimentional array that represents a given classification result
        uniformity_threshold (int): desired number of neighbours with the same class of the central pixel. This parameter accepts values from 1 to (kernel_size**2 - 1).
        kernel_size (int, optional): size of the moving window, used to create a square kernel structure composed of (kernel_size X kernel_size) pixels. Defaults to 3, which represents a 8-pixels neighbourhood.
        kernel_diagonals (bool, optional): indicates if diagonal values in the kernel object should be taken into consideration.
                    Useful to establish a 4-pixels neighbourhood. Defaults to True.
                    Example: If kernel_size is equal to 3 and kernel_diagonals is False, a 4-pixels neighbourhood is applied to compute the uniformity ratio.
                            Likewise, if kernel_size is equal to 5 and kernel_diagonals is False, only the 'pairs' in the cardinal positions (i.e., N, S, E and W) 
                                from the central pixel are used. In this example, 8 pixel values would be taken into consideration, instead of 24 pixels.
        target_class (int|str, optional): Class of interest. Defaults to None. 
                    In case no class of interest is informed, the index is computed to all the class labels identified in the input classification array.
                    Otherwise, the index is computed only to the requested class.
        border_mode (str): strategy to use with pixels that lie in the border of the array, as defined in the 'mode' parameter of the Scipy.generic_filter method. 
                Default is 'reflect', which extends the input array by reflecting the border pixels.
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html
        ignore_nan (bool, optional): Strategy to use with None/np.nan values. If True, None and np.nan are not considered as valid labels and ignored, and 0 is assigned to these pixels in the neighbourhood similarity matrix. Defaults to True.
        overall_key (str, optional): Key of the output dictionary that includes the overall statistics. Defaults to 'OVERALL'.
    
    Additional information:
        The 'footprint' parameter of the 'generic_filter' function can be defined as follows:
            - '4'-pixels neighbouring system (not only '4' neighbours if kernel_size > 3, but only cardinal positions are taken into consideration):
                m = np.zeros((kernel_size, kernel_size))
                m[int(m.shape[0]/2), :] = 1
                m[:, int(m.shape[0]/2)] = 1

            - 8-pixels neighbouring system:
                m = np.ones((kernel_size, kernel_size))

    Returns (Tuple):
        dict: dictionary with the number of occurrences and complete neighbourhood of each class, as well as the overall and per-class uniformity scores.
                The keys of the dictionary include the identified class labels (or only the requested target class) and the overall_key.
        np.array: neighbourhood similarity matrix. Each pixel of the matrix indicates the number of neighbours with the same class of the central pixel, considering the provided 'uniformity_threshold'.
    """    
    
    if isinstance(array, np.ndarray):
        if array.ndim != 2:
            raise ValueError(f"Provide a 2d np.ndarray. Try 'array.reshape(-1, array.shape[-1])' or 'array[0, :, :]' to reshape the data")
    else:
        raise ValueError(f"Provide a 2d np.ndarray.")
        
    # if diagonal pixels of the kernel structure should be analysed...
    if kernel_diagonals is True:
        # NOTE the central pixel does not count, only its neighbours
        n_neighbours = kernel_size**2 - 1
    else:
        m = np.zeros((kernel_size, kernel_size))
        m[int(m.shape[0]/2), :] = 1
        m[:, int(m.shape[0]/2)] = 1
        # NOTE the central pixel does not count, only its neighbours
        n_neighbours = m[m == 1].shape[0] - 1

    if uniformity_threshold > n_neighbours:
        uniformity_threshold = n_neighbours
        print(f"\n\nNOTE: uniformity threshold redefined to {uniformity_threshold}, in order to meet the number of neighbours.\n")


    def neighbourhood_similarity(array: np.array, kernel_size: int = 3, kernel_diagonals: bool = True, border_mode: str = 'reflect', ignore_nan: bool = True) -> np.array:
        """Return the neighbourhood similarity array of a given classification array
        
        Identifies the number of elements with the same value of the central element (i.e., neighbourhood similarity), from the provided list.

        Args:
            array (np.array): N-dimentional array that represents a given classification result
            kernel_size (int, optional): size of the moving window, used to create a square kernel structure composed of (kernel_size X kernel_size) pixels. Defaults to 3, which represents a 8-pixels neighbourhood.
            kernel_diagonals (bool, optional): indicates if diagonal values in the kernel object should be taken into consideration.
                        Useful to establish a 4-pixels neighbourhood. Defaults to True.
                        Example: If kernel_size is equal to 3 and kernel_diagonals is False, a 4-pixels neighbourhood is applied to compute the uniformity ratio.
                                Likewise, if kernel_size is equal to 5 and kernel_diagonals is False, only the 'pairs' in the cardinal positions (i.e., N, S, E and W) 
                                    from the central pixel are used. In this example, 8 pixel values would be taken into consideration, instead of 24 pixels.
            border_mode (str): strategy to use with border pixels, as defined in the Scipy.generic_filter method. Default is 'reflect', which extends the input array by reflecting the border pixels.
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html
            ignore_nan (bool, optional): Strategy to use with None/np.nan values. If True, None and np.nan are not considered as valid labels and 0 is returned. If False, the number of similar neighbours are computed like any other value. Defaults to True.

        Returns:
            np.array: neighbouhood similarity matrix, which indicates, for each pixel, the number of neighbouring pixels with the same class
        """
        def count_same_class_neighbours(invalues: list, ignore_nan: bool = True) -> int:
            """Return the number of elements in the provided list that present the same value of the central element
            
            Identifies the number of elements with the same value of the central element (i.e., neighbourhood similarity), from the provided list.

            Args:
                invalues (list): list of values extracted from a given position of the moving window
                ignore_nan (bool, optional): Strategy to use with None/np.nan values. If True, None and np.nan are not considered as valid labels and 0 is returned. If False, the number of similar neighbours are computed like any other value. Defaults to True.

            Returns:
                int: Number of elements with the same value of the central element
            """
            center_pos = int(invalues.shape[0]/2) #int((size**2)/2)
            center_val = invalues[center_pos]

            # TODO: verify if center_val is None or np.nan. Return 0 if center_val is None|np.nan and ignore_nan is True.
            if center_val in [None, np.nan]:
                if ignore_nan is True:
                    return 0
                
            # Subtracts the value related to the central pixel, used as reference
            num_identical_neighbours = np.count_nonzero(invalues == center_val) - 1
            return num_identical_neighbours

        #neigh_similarity_filter = lambda arr: generic_filter(arr, function=neighbourhood_similarity, size=kernel_size, mode='reflect', extra_arguments=((ignore_nan),))
        #neigh_similarity_filter = lambda arr: generic_filter(arr, function=neighbourhood_similarity, size=kernel_size, mode='reflect', extra_keywords={'ignore_nan':ignore_nan})

        # NOTE: conditional included only to change the 'generic_filter' parameters.
        #           Although size is automatically ignored if footprint is informed, an UserWarning is raised to informe that only footprint was used.
        if kernel_diagonals is True:
            neighbourhood_similarity_filter = lambda arr: generic_filter(arr, function=count_same_class_neighbours, size=kernel_size, mode=border_mode, extra_keywords={'ignore_nan':ignore_nan})
        else:
            ftp = np.zeros((kernel_size, kernel_size))
            ftp[int(kernel_size/2), :] = 1
            ftp[:, int(kernel_size/2)] = 1

            neighbourhood_similarity_filter = lambda arr: generic_filter(arr, function=count_same_class_neighbours, footprint=ftp, mode=border_mode, extra_keywords={'ignore_nan':ignore_nan})

        neighbourhood_similarity_matrix = neighbourhood_similarity_filter(array)

        return neighbourhood_similarity_matrix
    
    def class_uniformity(arr: np.array, neighbourhood_similarity_matrix: np.array, uniformity_threshold: int, 
                        target_class: int or str = None, ignore_nan: bool = True, overall_key: str = 'OVERALL',
                        ignore_img_borders:bool = True) -> dict:

        """Return a dictionary with the overall and per-class uniformity scores

        Compute the per-class and overall uniformity indices of a given classification result, based on the neighbourhood similarity matrix.

        Args:
            arr (np.array): N-dimentional array that represents a given classification result
            neighbourhood_similarity_matrix (np.array): N-dimentional array (like arr) that indicates the neighbourhood similarity of each pixel of the classification result (i.e., arr)
            uniformity_threshold (int): desired number of neighbours with the same class of the central pixel.
            target_class (int|str, optional): Class of interest. Defaults to None. 
                    In case no class of interest is informed, the index is computed to all the class labels identified in the input classification array.
                    Otherwise, the index is computed only to the requested class.
            ignore_nan (bool, optional): Strategy to use with None/np.nan values. If True, None and np.nan are not considered as valid labels and ignored. Defaults to True.
            overall_key (str, optional): Key of the output dictionary that includes the overall statistics. Defaults to 'OVERALL'.
        
        Returns:
            dict: dictionary with the number of occurrences and complete neighbourhood of each class, as well as the overall and per-class uniformity scores.
                The keys of the dictionary include the identified class labels (or only the requested target class) and the overall_key.
        """

        if ignore_img_borders is True:
            arr = arr[1:-1, 1:-1]
            neighbourhood_similarity_matrix = neighbourhood_similarity_matrix[1:-1, 1:-1]
            
        # Class(es) of interest
        if target_class is None:
            # remove None and np.nan, if any.
            if ignore_nan is True:
                classes = np.array([v for v in np.unique(arr) if v not in [np.nan, None]])
            else:
                classes = np.unique(arr)
        else:
            classes = np.array([target_class])

        uniformity_metrics = {}
        for class_ in classes:
            class_occurrences = np.count_nonzero(arr == class_)
            
            mask = ((arr==class_) & (neighbourhood_similarity_matrix >= uniformity_threshold)) * 1
            complete_neighbourhoods = np.count_nonzero(mask == 1)

            if class_occurrences > 0:
                class_uniformity_ratio = complete_neighbourhoods / class_occurrences
            else:
                class_uniformity_ratio = 0

            uniformity_metrics[class_] = {
                'occurrences': class_occurrences, 
                'complete_neighbourhoods': complete_neighbourhoods,
                'uniformity_ratio': round(class_uniformity_ratio, 4),
                'class_percentage': class_occurrences/arr.size,
            }
        
        #try:
        uniformity_metrics[overall_key] = {}
        uniformity_metrics[overall_key]['complete_neighbourhoods'] = sum([v['complete_neighbourhoods'] for (k, v) in uniformity_metrics.items() if str(k) in classes.astype(str)])
        uniformity_metrics[overall_key]['occurrences'] = sum([v['occurrences'] for (k, v) in uniformity_metrics.items() if str(k) in classes.astype(str)])
        
        ## simple mean
        #uniformity_metrics[overall_key]['simple_uniformity_ratio'] = round(
        #    uniformity_metrics[overall_key]['complete_neighbourhoods'] / uniformity_metrics[overall_key]['occurrences'],
        #    4
        #)

        # Weighted mean: the classes might not be equiprobable
        uniformity_metrics[overall_key]['uniformity_ratio'] = round(
            sum(
                [(v['complete_neighbourhoods']/v['occurrences'])*v['class_percentage'] for (k, v) in uniformity_metrics.items() if str(k) in classes.astype(str)]
            ),
            4
        )
        #except:
        #    print("Failed to convert class labels to str")
        #    uniformity_metrics[overall_key] = {}
        #    uniformity_metrics[overall_key]['complete_neighbourhoods'] = sum([v['complete_neighbourhoods'] for (k, v) in uniformity_metrics.items() if k in classes])
        #    uniformity_metrics[overall_key]['occurrences'] = sum([v['occurrences'] for (k, v) in uniformity_metrics.items() if k in classes])
        #    
        #    uniformity_metrics[overall_key]['simple_uniformity_ratio'] = round(
        #        uniformity_metrics[overall_key]['complete_neighbourhoods'] / uniformity_metrics[overall_key]['occurrences'],
        #        4
        #    )
        #
        #    # Weighted meam
        #    uniformity_metrics[overall_key]['uniformity_ratio'] = round(
        #        sum(
        #            [(v['complete_neighbourhoods']/v['occurrences'])*(v['occurrences']/arr.size) for (k, v) in uniformity_metrics.items() if str(k) in classes.astype(str)]
        #        ),
        #        4
        #    )

        return uniformity_metrics
    
    # (1) For each pixel, compute the number of neighbours assigned to its class
    neighbourhood_similarity_matrix = neighbourhood_similarity(array=array, kernel_size=kernel_size, kernel_diagonals=kernel_diagonals, border_mode=border_mode, ignore_nan=ignore_nan)

    # (2) Compute the classification uniformity index, including both the overall and per-class scores
    uniformity_metrics = class_uniformity(array, neighbourhood_similarity_matrix, uniformity_threshold, target_class, ignore_nan, overall_key=overall_key)

    return uniformity_metrics, neighbourhood_similarity_matrix
