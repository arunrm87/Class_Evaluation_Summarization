"""
Run using parse_doc.py <excel_file> <sheet_no>

(Currently works only for the dataset excel file at
 http://www.coursemirror.com/download/dataset/)
"""
import xlrd
import openpyxl
import sys

student_count = 52
point_of_interest = []
muddiest_point = []
learning_point = []

excel_file, sheet_no = sys.argv[1], int(sys.argv[2])

## REMEMBER TO ADD "None" as a stopword, apart from ENGLISH stopwords

workbook = openpyxl.load_workbook(excel_file)
sheet_names = workbook.get_sheet_names()
sheet = workbook.get_sheet_by_name(sheet_names[sheet_no])

for row in range(3,student_count + 3):
	point_of_interest.append(sheet['B' + str(row)].value)
	muddiest_point.append(sheet['C' + str(row)].value)
	learning_point.append(sheet['D' + str(row)].value)

outputFile_interest = open('output%s_interest' % sheet_no, 'w')
outputFile_muddiest = open('output%s_muddiest' % sheet_no, 'w')
outputFile_learning = open('output%s_learning' % sheet_no, 'w')

for row in range(0,len(point_of_interest)):
	if (point_of_interest[row] == None):
		outputFile_interest.write("None")
	else:
		outputFile_interest.write(point_of_interest[row])
	outputFile_interest.write("\n")

	if (muddiest_point[row] == None):
		outputFile_muddiest.write("None")
	else:
		outputFile_muddiest.write(muddiest_point[row])
	outputFile_muddiest.write("\n")

	if (learning_point[row] == None):
		outputFile_learning.write("None")
	else:
		outputFile_learning.write(learning_point[row])
	outputFile_learning.write("\n")

