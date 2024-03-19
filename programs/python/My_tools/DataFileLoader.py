import copy, re, os
import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
from collections import deque
from multiprocessing import Pool
import textwrap as tw
import warnings
import pickle
import logging

class DataFileLoader:
    """
        DataFileLoader stores files in a hashmap with keys given by the file name of the file spesified by the `file_pattern`.
    """
    def __init__(self, data_path, file_pattern, key_name = None,_iter_key = False):
        r"""
        Initializes the DataFileLoader class.

        Args:
            data_path (str): Path to the directory containing data files.
            file_pattern (str): File name pattern writen in RegEx notation spesified by the python `re` module to match files. The file pattern should contain capture groups to signify the keys used for groupings.
            key_names (function): Converts RegEx groups to a different string based on a user defined function that takes one argument of type list of strings.
                Example: r"file_y\d{4}_id\d{1,3}.csv"
        """
        self.data_path = data_path
        self.file_pattern = file_pattern
        self.pt = re.compile(self.file_pattern)
        self.DictData = {}
        self._origin = None
        self.levels = self.pt.groups
        self._nolevCount = None
        self.grouped = False
        self.key_name = key_name
        self._iter_key = _iter_key
        self._leaf_list = []

    def load_data(self, group_order = None, names = None):
        """
        Loads data files from the specified path and populates the nested dictionary.

        Args:
            group_order (list): Spesifies the order of the keys.
            names (list): Spesifies the names of columns in files.
        """
        logging.info("started load")
        file_readers = {
            "csv": self._read_csv,
            "xlsx": self._read_xlsx,
            # Add other formats as needed
        }
        for root, _, files in os.walk(self.data_path):
            for filename in [filename for filename in files if self.pt.match(filename)]:
                try:
                    filepath = os.path.join(root, filename)
                    identifier = self._extract_placeholders(filename)
                    #identifier.reverse()  # Stack-based filling of structure
                    if group_order:
                        identifier = [identifier[i] for i in group_order]
                    if callable(self.key_name):
                        identifier = self.key_name(identifier)
                    if identifier:
                        file_extension = filename.split(".")[-1]

                        if file_extension in file_readers:
                            df = file_readers[file_extension](filepath,names)
                        else:
                            logging.error(f"Unsupported file type: {file_extension}")
                            continue

                        # Update the nested dictionary
                        current_dict = self.DictData
                        for k in identifier[:-1]:
                            current_dict = current_dict.setdefault(k, {})
                        current_dict[identifier[-1]] = df
                        self._leaf_list.append((identifier[-1], df) if self._iter_key else df)

                except Exception as e:
                    logging.error(f"Error loading data from {filename}: {e}")
        logging.info("ended load")

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
    
    def _extract_placeholders(self, filename) -> list[str]:
        """
        Extracts placeholders from the filename based on the specified pattern.

        Args:
            filename (str): File name.

        Returns:
            List[string]: Extracted year and identifier (or None if not found).
        """

        if self.levels == 0:
            if self._nolevCount is None:
                self._nolevCount = 0
            self._nolevCount += 1
            return self._nolevCount

        try:
            placeholders = list(self.pt.findall(filename)[0])
            #print(placeholders)
            return placeholders
        except (ValueError, IndexError):
            return None
        
    def find_depth(self, Dict ,_current_depth = 0):
        if isinstance(Dict,dict):
            return _current_depth + max(map(lambda x: self.find_depth(x,_current_depth+1),Dict.values()))
        return _current_depth

    def group_layer(self,group_struct: dict[str], level = None):
        """
            Groups a layer in accordence with `group_struct` which is a dictionary.

            Description:
                This function will look through the structure and group the entries in accordence with the scematic given by the `group_struct`
            
            Args:
                group_struct (dict[string]): grouping schematic
                level (int | NoneType): The depth to group, if None the a search will procead

            Returns:
                DataFileLoader: Restructured DataFileLoader
        """

        new_data = copy.deepcopy(group_struct)

        old_data = copy.deepcopy(dict(self.flatten(max_level = level,return_key=True))) if level else copy.deepcopy(self.DictData)

        Stack = deque([new_data])

        current_list = {}
        add_levs = self.find_depth(new_data)

        while len(Stack) > 0:
            current_list = Stack.popleft()
            if isinstance(list(current_list.values())[0], list):
                #print(current_list.items(), old_data.keys())
                for key, surrugate_keys in current_list.items():
                    current_list[key] = {surr_key: old_data[surr_key] for surr_key in surrugate_keys}
            else:
                Stack.append(current_list.values())

        new_ret = self._dict2class(new_data)
        new_ret.levels += add_levs

        new_ret.grouped = True

        return new_ret

    def _update_leafs(self):
        self._leaf_list = copy.deepcopy(self.flatten(return_key=self._iter_key,_force_update = True))

    def shave_top_layer(self):
        """
            Removes top layer of the structure and merges child-trees to a single structure.
        """
        gathered_data = {}
        for value in self.DictData.values():
            gathered_data.update(value)
        new_ret = self._dict2class(gathered_data,params={"levels":self.levels-1})
        return new_ret

    def merge_layer(self, level = 0, merge_func = None, _current_level = 0):
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
        DictData = copy.deepcopy(self.DictData)
        if _current_level == level:
            if merge_func:
                return self.flatten(merge_func = merge_func)
            else:
                return self.flatten(merge_func=lambda x,y: pd.concat([x, y], ignore_index=True))
        else:
            current_dict = {}
            for key in DictData.keys():
                if isinstance(DictData[key], dict):
                    current_dict.update({key: self._dict2class(DictData[key]).merge_layer(level = level, merge_func = merge_func, _current_level = _current_level + 1)})
                else:
                    raise ValueError("level too large, current level = "+str(level))
            return self._dict2class(current_dict)

    def flatten(self, merge_func = None, max_level = None, return_key = False, _current_level = 0, _force_update = False):
        """
        Flattens the entire datastructure to a single DataFrame

        Args:
            merge_func (function(x,y)): Function to merge all data, needs to take two arguments

        Returns:
            data DataFrame: Transformed data
        """
        if (merge_func is None) and (max_level is None) and (self._iter_key == return_key) and not(_force_update):
            return self._leaf_list
        
        #print(_current_level)

        return_value = None
        return_keys = []
        for key, value in self.DictData.items():
            #should_process_value = isinstance(value, dict) and ((not max_level) or (_current_level != max_level))

            #processed_value = (self._dict2class(value).flatten(merge_func=merge_func, max_level=max_level, _current_level=_current_level+1) if should_process_value else value) if merge_func else [value]

            #return_value = merge_func(return_value, processed_value) if merge_func and return_value is not None else (return_value + processed_value if return_value else processed_value)

            #if return_key:
            #    return_keys.append(key)
            if isinstance(value, dict) and ((_current_level != max_level) if max_level else True):
                #print("inside, current level", _current_level)
                processed_value = self._dict2class(value).flatten(merge_func = merge_func, max_level = max_level, _current_level = _current_level+1)
            else:
                processed_value = value if merge_func else [value]
            if return_value is not None:
                merged_value = merge_func(return_value, processed_value) if merge_func else processed_value
                if return_key:
                    return_value.append(processed_value)
                    return_keys.append(key)
                else:
                    return_value = (return_value + processed_value)
            else:
                if return_key:
                    return_value = [processed_value]
                    return_keys.append(key)
                else:
                    return_value = processed_value

        return list(zip(return_keys,return_value)) if return_key else return_value

    def data_transform(self,func = None, clone=True):
        """
        Transforms data via a function that take in a dataframe and turns it into another dataframe

        Args:
            func (function): Function to transform data

        Returns:
            data DataFileLoader: Transformed data
        """
        if clone:
            data = copy.deepcopy(self)
        else:
            data = self
        if func is None:
            return
        for key, value in data.DictData.items():
            if isinstance(value, dict):
                self._dict2class(value).data_transform(func,clone=False) # only need to clone on top, not the entire tree
            else:
                data.DictData[key] = func(value)
        return data

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

    def _dict2class(self, object, params = None):
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
        ret._update_leafs()
        if params is not None:
            ret.__dict__.update(params)
        return ret

    def combine(self, other, keep_levs = None,merge_func = None):
        assert(self.levels == other.levels)
        
        data = copy.deepcopy(self.DictData)
        other_dict = copy.deepcopy(other.DictData)

        params = {"levels": self.levels} if keep_levs or (self.grouped and keep_levs is None) else None
        
        common = data.keys() & other_dict.keys()
        not_common = data.keys() ^ other_dict.keys()

        for key in common:
            if isinstance(data[key],dict) & isinstance(other_dict[key],dict):
                com_data = self._dict2class(data[key])
                com_other = other._dict2class(other_dict[key])
                merged = com_data.combine(com_other,merge_func = merge_func)
                data[key] = merged.DictData
            elif merge_func is not None:
                data[key] = merge_func(data[key],other_dict[key])
            else:
                data[key] = pd.concat([data[key], other_dict[key]], ignore_index=True)
        
        data.update({key: other_dict[key] for key in not_common})
        
        new_ret = self._dict2class(data,params=params)
        return copy.deepcopy(new_ret)
    
    def clone(self):
        return copy.deepcopy(self)
    
    def __getitem__(self, indexes): 
        current_dict = {}

        if isinstance(indexes, str):
            if indexes in self.DictData:
                current_dict = {indexes : self.DictData[indexes]}
            else:
                raise KeyError(str(indexes) + " is not in data.")
        elif isinstance(indexes, tuple):
            #assert self.levels == len(indexes), "max level is {} but got {}".format(self.levels,len(indexes))
            if len(indexes) == 0:
                return self
            
            temp = self[indexes[0]]

            rest_ind = tuple(list(indexes)[1:])

            for key, value in temp.DictData.items():
                if isinstance(value,dict):
                    psudo_data = self._dict2class(value,params= {"levels":self.levels-1})
                    psudo_data_dict = psudo_data[rest_ind].DictData
                    temp.DictData[key] = psudo_data_dict
            
            current_dict = temp.DictData

        elif isinstance(indexes,list):
            current_dict = {i:self.DictData[i] for i in indexes}

        elif isinstance(indexes,slice):
            values_sorted = sorted(self.DictData.keys())

            if indexes.start not in values_sorted or indexes.stop not in values_sorted:
                missing = [index for index in (indexes.start, indexes.stop) if (index not in values_sorted) and (index is not None)]
                if len(missing)>0:
                    raise KeyError("Index not in data: " + str(missing), "Avalible indexes: " + str(values_sorted))
            
            ind_start = values_sorted.index(indexes.start) if indexes.start in values_sorted else 0
            ind_end = values_sorted.index(indexes.stop) + 1 if indexes.stop in values_sorted else len(values_sorted)

            if (indexes.step is not None) & isinstance(type(indexes.step),int):
                ind = values_sorted[ind_start:ind_end:indexes.step]
            else:
                ind = values_sorted[ind_start:ind_end]
            current_dict = self[ind].DictData
        
        return copy.deepcopy(self._dict2class(current_dict))
    
    def dump(self,FileName):
        """
            Saves class to a binary file named `FileName`
        """
        pickle.dump(self,f := open(FileName,"wb"))
        f.close()

    def __str__(self) -> str:
        ret_string = ""
        for key, value in self.DictData.items():
            if isinstance(value,dict):
                str_value = str(self._dict2class(value))
                ret_string += str(key) + ":\n" + tw.indent(str_value,"\t") + "\n\n"
            else:
                ret_string += str(key) + ":\n" + tw.indent(str(value),"\t") + "\n\n"
        return ret_string
    
    def __iter__(self):
        return self._leaf_list.__iter__()

class ExperimentDataFileHandler(DataFileLoader):
    
    def __init__(*args):
        warnings.warn("This an experimental class, features might give wrong values or worse.",RuntimeWarning)
        super().__init__(*args)
        
    def _data_prosess(self,file_info):
        ext_to_reader = {
            "csv": self._read_csv,
            "xlsx": self._read_xlsx,
            # Add other formats as needed
        }
        obj, filename, filepath, identifier, file_extension, names = file_info
        try:
            reader_func = ext_to_reader.get(file_extension)
                            
            if reader_func:
                df = reader_func(filepath, names=names)
            else:
                raise TypeError(f"Unrecognized file type: {file_extension}")
            return (identifier,df)
        except Exception as e:
            logging.warning(f"Error loading data from {filename}: {e}")

    def load_data(self, group_order=None, names=None):
        """
        Loads data files from the specified path and populates the nested dictionary.
        """
        logging.info("started load")

        File_info_list = []
        for root, _, files in os.walk(self.data_path):
            for filename in [filename for filename in files if self.pt.match(filename)]:

                filepath = os.path.join(root, filename)
                try:
                    identifier = self._extract_placeholders(filename)
                    if identifier:
                        identifier.reverse()  # stack-based filling of structure   
                        if group_order:
                            identifier = [identifier[i] for i in group_order]
        
                        file_extension = filename.split(".")[-1]
                        File_info_list.append((self,filename, filepath, identifier, file_extension,names))

                except Exception as e:
                    logging.warning(f"Error loading data from {filename}: {e}")
        num_processes = os.cpu_count()

        for data in Pool(num_processes).map(self._data_prosess, File_info_list):
            current_dict = self.DictData
            identifier = data[0]
            while len(identifier) > 1:
                k = identifier.pop()
                current_dict.setdefault(k, {})
                current_dict = current_dict[k]
            
            current_dict[identifier[0]] = data[1]
        logging.info("ended load")

    def _dict2class(self, object):
        """
            Restorerer dataen til en forandring tilbake.

            Args:
                object Dict: object to be converted
            
            Return:
                DataFileLoader
        """
        ret = ExperimentDataFileHandler(self.data_path, self.file_pattern)
        ret.DictData = object
        ret._origin = self
        return ret
            

if __name__ == "__main__":
    pass
