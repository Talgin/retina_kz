import datetime

print(str(datetime.datetime.now()).replace(":", "_").replace(".", "_").replace("-", "_").replace(' ', '_'))