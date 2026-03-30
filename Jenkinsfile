pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo "Checking out code from GitHub..."
                checkout scm
            }
        }
        
        stage('Train Model') {
            steps {
                echo "Training Credit Risk Model..."
                sh '''
                    cd $WORKSPACE
                    python3 src/train_v2.py
                '''
            }
        }
        
        stage('Test API') {
            steps {
                echo "Testing API Health..."
                sh '''
                    curl -f http://host.docker.internal:8000/health
                '''
            }
        }
    }
    
    post {
        success {
            echo "Pipeline Successful!"
        }
        failure {
            echo "Pipeline Failed!"
        }
    }
}
