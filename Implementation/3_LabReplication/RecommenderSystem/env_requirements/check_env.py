import pkg_resources
import required_libs as l
import subprocess

#Check you environment and install missing packages

installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
my_libs = []
for value in installed_packages_list:
    my_libs.append(value)

diff = [pac for pac in l.required_libs if pac not in my_libs]

if len(diff) > 0:
    print(diff)
    while True:
        r = input("Type 'ok' to install missing packages?").lower()
        if r == 'ok':
            print('Installing...')
            for lib in diff:
                subprocess.check_call(['pip', 'install', lib])
            break
        else:
            break
else:
    print("No differences found")