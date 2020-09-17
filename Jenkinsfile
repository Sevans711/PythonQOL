/* comment test */

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
