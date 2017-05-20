#!/usr/bin/env python3

import zipfile
import os
import traceback


def packager(submission_dir, student_py_file, student_pdf_file):
    """
    check the syntactic validity of student's submitted project3.py file and
    the completeness of the methods and classes
    """
    os.chdir(submission_dir)
    with open(student_py_file) as f_sub:
        src = f_sub.read()
        globals = {}
        exec(compile(src, student_py_file, 'exec'), globals)

    class_list = ['MixtureModel', 'GMM', 'CMM']
    method_list = ['e_step', 'm_step', 'fit']

    for c in class_list:
        assert c in globals, 'missing class %s' % c
        for m in method_list:
            assert m in dir(globals[c]), \
                    'class %s is missing method %s' % (c, m)

    with zipfile.ZipFile('submit.zip', 'w') as zf:
        for filename in [student_py_file, student_pdf_file]:
            data = zf.write(filename)

    with zipfile.ZipFile('submit.zip', 'r') as zf:
        for filename in [student_py_file, student_pdf_file]:
            data = zf.read(filename)

if __name__ == '__main__':
    student_py_file = 'project3.py'
    student_pdf_file = 'writeup.pdf'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    submission_dir = cur_dir
    packager(submission_dir, student_py_file, student_pdf_file)
    print('Successfully created submit.zip!')
