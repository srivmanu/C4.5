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
print "\n\nTEST on 20% data\n\n" 
c1.testNode()
print "\n\nK-Cross : \nk = 3\n\n"
c1.kCross(3)
# print "TEST RESULT"
# c1.testNode()
