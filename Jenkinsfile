pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo '📦 Pulling code from GitHub...'
                checkout scm
            }
        }
        
        stage('Train Model') {
            steps {
                echo '🤖 Training Credit Risk Model...'
                bat '''
                    cd C:\\Users\\siddh\\Desktop\\credit-risk-mlops
                    docker-compose exec dev python src/train_v2.py
                '''
            }
        }
        
        stage('Test API') {
            steps {
                echo '🔍 Testing API...'
                bat 'curl -f http://localhost:8000/health'
            }
        }
    }
    
    post {
        success {
            echo '🎉 Pipeline Successful!'
        }
        failure {
            echo '❌ Pipeline Failed!'
        }
    }
}
