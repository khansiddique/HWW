# if '__main__'=="name":
#   print('ok')
#   # pass
import TestDataProtocolExchange.SignalManagementPython.SignalManagementPython as hello
import TestDataProtocolExchange.SignalPreparation.PrepareDataSet as prep


# import hello

hello.my_function()

print(hello.name)

nicholas = hello.Signal()
nicholas.get_student_details()

# def classcall():
#   c= SMP.Signal()
#   s= SMP.test()
  
def prepdataset():
  prep.preptest()
  print('call prapare dataset function test()')
if __name__ == '__main__':
  # classcall()
  prepdataset()
  print('ok')