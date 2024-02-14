import copy, re, os
import pandas as pd
class DataFileLoader:
    def __init__(self, data_path, file_pattern):
        """
        Initializes the DataFileLoader class.

        Args:
            data_path (str): Path to the directory containing data files.
            file_pattern (str): File name pattern (with placeholders) to match files.
                Example: r"file_y\d{4}_id\d{1,3}.csv"
        """
        self.data_path = data_path
        self.file_pattern = file_pattern
        self.pt = re.compile(self.file_pattern)
        self.DictData = {}
        self._origin = None
        self.levels = self.pt.groups

    def load_data(self, group_order = None, names = None):
        """
        Loads data files from the specified path and populates the nested dictionary.
        """
        print("started load")
        for root, _, files in os.walk(self.data_path):
            for filename in files:
                if not self.pt.match(filename):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    # Extract placeholders from the filename using regex or string manipulation
                    # For example, if the filename is "file_y2045_id190.csv", extract 2045 and 190
                    # and convert them to integers
                    identifier = self._extract_placeholders(filename)
                    identifier.reverse() # stack based filling of structure
                    if group_order is not None:
                        identifier = [identifier[i] for i in group_order]
                    if len(identifier) > 0 and identifier is not None:
                        # Read the CSV file into a Pandas DataFrame
                        file_extention = filename.split(".")[-1]

                        match file_extention:
                            case "csv":
                                df = self._read_csv(filepath, names = names)
                            case "xlsx":
                                df = self._read_xlsx(filepath, names = names)
                            case "json":
                                raise NotImplementedError("Have not implemented Json")
                            case _:
                                raise TypeError("Did not reconice type: " + file_extention)
                        
                        # Update the nested dictionary
                        current_dict = self.DictData
                        while len(identifier) > 1: # to make it more general
                            k = identifier.pop()
                            if k not in current_dict:
                                current_dict[k] = {}
                            current_dict = current_dict[k]
                        current_dict[identifier[0]] = df
                except Exception as e:
                    print(f"Error loading data from {filename}: {e}")
        print("ended load")

    def _read_csv(self, file_path, names = None):
        df = pd.read_csv(file_path,delimiter = ",",
                                   header = 0,
                                   names = names)
        df.iloc[:,0] = pd.to_datetime(df.iloc[:,0] + "00",
                                    format = "%Y-%m-%d %H:%M:%S%z",
                                    utc=True) # converts summer time to timezone then convert to GMT
        return df

    def _read_xlsx(self, file_path, names = None):
        df = pd.read_xlsx(file_path)
    
    def _extract_placeholders(self, filename):
        """
        Extracts placeholders from the filename based on the specified pattern.

        Args:
            filename (str): File name.

        Returns:
            List[string]: Extracted year and identifier (or None if not found).
        """
        try:
            placeholders = list(self.pt.findall(filename)[0])
            #print(placeholders)
            return placeholders
        except (ValueError, IndexError):
            return None

    def merge_layer(self, level = 0, _current_level = 0, merge_func = None):
        """
        Flattens a given layer in a 0 indexed way.

        Example:
            Assuming the leaf is a list of int
            {'a':{'1':[5]},'b':{'1':[6],'2':[1,6]}} -> .merge_layer(level = 1) -> {'a':[5],'b':[6,1,6]}
            {'a':{'1':5},'b':{'1':6,'2':1}} -> .merge_layer(level = 0) -> [5,6,1,6]

        Args:
            merge_func (function(x,y)): Function to merge all data, needs to take two arguments

        Returns:
            data DataFrame: Transformed data
        """
        if _current_level == level:
            return self.flatten(merge_func = merge_func)
        else:
            current_dict = {}
            for key in self.DictData.keys():
                if isinstance(self.DictData[key], dict):
                    current_dict.update({key: self._dict2class(self.DictData[key]).merge_layer(level,_current_level + 1, merge_func = merge_func)})
                else:
                    raise ValueError("level too large, current level = "+str(level))
            return self._dict2class(current_dict)

    def flatten(self, merge_func = None):
        """
        Flattens the entire datastructure to a single DataFrame

        Args:
            merge_func (function(x,y)): Function to merge all data, needs to take two arguments

        Returns:
            data DataFrame: Transformed data
        """
        data = copy.deepcopy(self.DictData)
        return_value = None
        stack = list(data.values())
        while len(stack) > 0:
            current_dict = stack.pop()
            if not(isinstance(current_dict,dict)):
                if (return_value is not None) and (merge_func is None):
                    return_value = pd.concat([return_value, current_dict], ignore_index=True)
                elif return_value is not None:
                    return_value = merge_func(return_value, current_dict)
                else:
                    return_value = current_dict
                continue
            for value in current_dict.values():
                if isinstance(value,dict):
                    stack.extend(list(value.values()))
                else:
                    if (return_value is not None) and (merge_func is None):
                        return_value = pd.concat([return_value, value], ignore_index=True)
                    elif return_value is not None:
                        return_value = merge_func(return_value, value)
                    else:
                        return_value = value
        return return_value

    def data_transform(self,func):
        """
        Transforms data via a function that take in a dataframe and turns it into another dataframe

        Args:
            func (function): Function to transform data

        Returns:
            dict Dict[int][int]DataFrame: Transformed data
        """
        data = copy.deepcopy(self.DictData)
        stack = list(data.values())
        while len(stack) > 0:
            current_dict = stack.pop()
            for key, value in current_dict.items():
                if isinstance(value,dict):
                    stack.extend(list(value.values()))
                else:
                    current_dict[key] = func(current_dict[key]) # apply function to all leafs
        return self._dict2class(data)
    def restore(self):
        """
            Restorerer dataen til en forandring tilbake.
            
            Return:
                DataFileLoader
        """
        if self._origin is None:
            return self
        else:
            return self._origin

    def _dict2class(self, object):
        """
            Restorerer dataen til en forandring tilbake.

            Args:
                object Dict: object to be converted
            
            Return:
                DataFileLoader
        """
        ret = DataFileLoader(self.data_path, self.file_pattern)
        ret.DictData = object
        ret._origin = self
        return ret

    def combine(self, other, merge_func = None):
        assert(self.levels == other.levels)
        
        data = copy.deepcopy(self.DictData)
        other_dict = copy.deepcopy(other.DictData)
        
        common = set(data.keys()) & set(other_dict.keys())
        not_common = (set(data.keys()) & set(other_dict.keys())) - common

        for key in common:
            if isinstance(data[key],dict) & isinstance(other_dict[key],dict):
                com_data = self._dict2class(data[key])
                com_other = other._dict2class(other_dict[key])
                merged = com_data.combine(com_other,merge_func = merge_func)
                data[key] = merged.DictData
            elif merge_func is not None:
                merged = merge_func(data[key],other_dict[key])
                data[key] = merged
            else:
                merged = pd.concat([data[key], other_dict[key]], ignore_index=True)
                data[key] = merged

        for key in not_common:
            data.update({key:other_dict[key]})
        
        return self._dict2class(data)
    
    def __getitem__(self, indexes): #! Does not work!
        current_dict = self.DictData
        if isinstance(indexes,tuple):
            for i in indexes:
                if isinstance(i,tuple | list | slice):
                    current_dict = self._dict2class(current_dict)[i].DictData
                else:
                    current_dict = current_dict[i]
        elif isinstance(indexes,list):
            ret = {}
            for i in indexes:
                if isinstance(i,tuple | list | slice):
                    current_dict = self._dict2class(current_dict)[i].DictData
                else:
                    ret.update({i:current_dict[i]})
            current_dict = ret
        elif isinstance(indexes,slice):
            # check if in dict
            sub_dict = {}
            values_sorted = sorted(current_dict.keys())
            
            if (indexes.start in values_sorted) & (indexes.stop in values_sorted):
                ind_start = values_sorted.index(indexes.start)
                ind_end = values_sorted.index(indexes.stop)
            elif (indexes.start is None) | (indexes.stop is None):
                ind_start = values_sorted.index(indexes.start) if indexes.start in values_sorted else 0
                ind_end = values_sorted.index(indexes.stop) if indexes.stop in values_sorted else len(values_sorted)-1
            else:
                raise ValueError("confused key pair: ({},{}) ; known keys:{}".format(indexes.start,indexes.stop,values_sorted) )

            if (indexes.step is not None) & isinstance(type(indexes.step),int):
                ind = values_sorted[ind_start:ind_end:indexes.step]
            else:
                ind = values_sorted[ind_start:ind_end]
            for i in ind:
                sub_dict.update({i:current_dict[i]})
            current_dict = sub_dict
        else:
            current_dict = self.DictData[indexes]
        if isinstance(current_dict,dict):
            ret_dict = self._dict2class(current_dict)
        else:
            return current_dict

if __name__ == "__main__":
    pass