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
                    echo "Workspace path: $WORKSPACE"
                    ls -la $WORKSPACE/src/
                    docker run --rm -v $WORKSPACE:/workspace -w /workspace python:3.9-slim sh -c "pip install mlflow scikit-learn numpy pandas && ls -la /workspace/src/ && python /workspace/src/train_v2.py"
                '''
            }
        }
        
        stage('Test API') {
            steps {
                echo "Testing API Health..."
                sh '''
                    curl -f http://localhost:8000/health
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
