import copy
import csv 
#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            tuple of int: rows, cols in the table
        Notes:
            Raise ValueError on invalid col_identifier
        """
        #empty list to fill data into the column
        column = []
        for line in self.data:
            # if the col_identifier does't exist in the table raise ValueErrror
            try:
                idx = self.column_names.index(col_identifier)
            except:
                raise ValueError

            if include_missing_values:
                column.append(line[idx])
            else:
                if line[idx] != "NA":
                    column.append(line[idx])

        return column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
            ugly solution
        """
        for x, line in enumerate(self.data):
            for y, col in enumerate(line):
                try:
                    self.data[x][y] = float(col)
                except ValueError as empty:
                    pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.
        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        for row in rows_to_drop:
            if self.data.index(row):
               self.data.remove(row) 

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename) as f:
            csv_obj = csv.reader(f, delimiter=',')
            header = True
            for line in csv_obj:
                if header:
                    self.column_names = line
                    header = not header
                else:
                    self.data.append(line)
        self.convert_to_numeric() 

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as f:
            csv_obj = csv.writer(f)
            csv_obj.writerow(self.column_names)
            csv_obj.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicates = []
        non_duplicates = []
        data = []
        # get the column(s)' data of the table using get_column func
        for key in key_column_names:
            data.append(self.get_column(key))
        # store data in a temp var which now should have the rows and cols in the table of the column
        temp = [*zip(*data)]

        for counter, item in enumerate(temp):
            if not item in non_duplicates:
                non_duplicates.append(item)
            else:
                duplicates.append(self.data[counter])
        return duplicates


    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
         # iterate through each row and check if the row contain NA using index built in func
        for row in self.data:
            try:
                if "NA" in row:
                    self.data.remove(row)
            except ValueError:
                pass

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        
        count = 0
        total = 0
        index = self.column_names.index(col_name)
        for line in self.data:
            if line[index] != "NA":
                total += line[index]
                count += 1

        average = total/count

        for line in self.data:
            if line[index] == "NA":
                line[index] = average


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. 
                The column names and their order is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        data = []
        for name in col_names:
            col = self.get_column(name, include_missing_values=False)

            if col:
                attribute = name
                minm = min(col)
                maxm = max(col)
                mid = maxm - ((maxm - minm) / 2)
                avg = sum(col)/len(col)

                col.sort()
                midl = len(col) // 2
                median = (col[midl] + col[~midl]) / 2.0

                data.append([attribute, minm, maxm, mid, avg, median])

        return MyPyTable(["attribute", "min", "max", "mid", "average", "median"], data) 


    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table = []
        key1 = []
        key2 = []

        header = self.column_names
        for x in other_table.column_names:
            if x not in key_column_names:
                header.append(x)
        for key in key_column_names:
            key1.append(self.column_names.index(key))
            key2.append(other_table.column_names.index(key))
        for idx, row in enumerate(self.data):
            for idx2, row2 in enumerate(other_table.data):
                join = True
                try:
                    for k, k1 in enumerate(key1):
                        if row[k1] != row2[key2[k]]:
                            join = False
                            raise Exception
                except:
                    pass
                if join:
                    new = copy.deepcopy(row)
                    for a, b in enumerate(row2):
                        if a not in key2:
                            new.append(b)
                    joined_table.append(new)

        return MyPyTable(header, joined_table)


    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_columne fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        # add both table header to the header of the new table
        header = self.column_names
        for _,j in enumerate(other_table.column_names):
            if j not in header:
                header.append(j)

        joinedData = self.data
        currKeyIndex = []
        otherKeyIndex = []
        rowsAdded = []
        for cols in key_column_names:
                currKeyIndex.append(self.column_names.index(cols))
                otherKeyIndex.append(other_table.column_names.index(cols))

        for rows in range(len(joinedData)):
            for other_rows in range(len(other_table.data)):
                if len(key_column_names) == 1:  
                    if other_table.data[other_rows][0] in joinedData[rows][0]:
                        fillData  = otherKeyIndex[-1]
                        rowsAdded.append(other_rows)
                        for cols_num in range(fillData+1, len(other_table.column_names)):
                            joinedData[rows].append(other_table.data[other_rows][cols_num])
                else :
                    matched = True
                    for c in range(len(key_column_names)):
                        if other_table.data[other_rows][otherKeyIndex[c]] != joinedData[rows][currKeyIndex[c]]:
                            matched = False
                    if matched:
                        fillData  = otherKeyIndex[-1]
                        rowsAdded.append(other_rows)
                        for cols_num in range(fillData+1, len(other_table.column_names)):
                            joinedData[rows].append(other_table.data[other_rows][cols_num])
            if len(joinedData[rows]) != len(header):
                fillData = len(header) - len(joinedData[rows])
                for cols_num in range(fillData):
                    joinedData[rows].append("NA")
                                
        # add the remaining instances using the key index to compare        
        for other_rows in range(len(other_table.data)):
            if other_rows not in rowsAdded:
                #fill in NA if need to be then add the element
               joinedData.append(["NA" for i in range(len(header))])
               for idx, cols in enumerate(header):
                   if cols in other_table.column_names:
                       joinedData[len(joinedData)-1][idx] = other_table.data[other_rows][other_table.column_names.index(cols)]
        return MyPyTable(header, joinedData) 