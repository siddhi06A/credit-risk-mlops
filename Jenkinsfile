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
                    echo "Jenkins Workspace: $WORKSPACE"
                    docker run --rm -v $WORKSPACE:$WORKSPACE -w $WORKSPACE python:3.9-slim sh -c "pip install mlflow scikit-learn numpy pandas && ls -la $WORKSPACE/src/ && python $WORKSPACE/src/train_v2.py"
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
