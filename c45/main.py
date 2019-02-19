from c45 import C45
import sys

args = sys.argv
if (len(args) != 3):
    print("Either Missing data OR Extra Data provided")
    exit()
data_file_name = args[1]
name_file_name = args[2]
c1 = C45(data_file_name,name_file_name)
c1.fetchData()
c1.preprocessData()
c1.generateTree()
c1.printTree()

# print "TEST RESULT"
# c1.testNode()
