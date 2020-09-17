/*
If you don't know what this file is, you can ignore it completely.
It has nothing to do with making the PythonQOL code work.
It is related making it easier to test or use PythonQOL in a code pipeline.
*/

pipeline {
   agent any
   stages {
      stage('update') {
         steps {
            sh '''#!/bin/csh 
            echo hello jenkins
            echo $PATH
'''
         }
      }
         stage('installing') {
         steps {
            sh '''#!/bin/csh 
            pip install -e .
'''
         }
      }
         stage('create test py') {
         steps {
            sh '''#!/bin/csh 
            cat <<-'TEST_CASES' > test.py
#!/usr/bin/env python
import QOL.files as fqol
import QOL.plots as pqol
'''
         }
      }
      stage('test import') {
         steps {
            sh '''#!/bin/csh 
            chmod +x test.py
            ./test.py 
            exit	  
'''
         }
      }
   }
}
