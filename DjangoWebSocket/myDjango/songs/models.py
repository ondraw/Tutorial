from django.db import models

class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
    
    #Question.objects.all() 의 결과가 <QuerySet [<Question: Question object (1)>]> 나온다. 이것을
    #텍스트로 나오게 하려면 __str__을 확장해준다.
    def __str__(self): #
        return self.question_text


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)
    
    def __str__(self):
        return self.choice_text