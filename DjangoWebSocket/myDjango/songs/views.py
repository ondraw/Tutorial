from django.shortcuts import render,get_object_or_404
from django.template.context_processors import request
from django.template import loader
from django.http.response import HttpResponse
from django.http import Http404,HttpResponseRedirect
from django.urls import reverse
from django.views import generic

from .models import Question,Choice


def index(request):
    #질문 객체의 pub_date로 정렬 한 후 리스트 0~5까지만 보여준다.
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    
    template = loader.get_template('index.html')
    context = {'latest_question_list': latest_question_list}
    return HttpResponse(template.render(context, request))

    #output = ', '.join([q.question_text for q in latest_question_list])
    #return HttpResponse(output)



def detail(request, question_id):
#     try:
#         question = Question.objects.get(pk=question_id)
#     except Question.DoesNotExist:
#         raise Http404("질문 페이지가 존재 하지 않습니다.")
#     return render(request, 'detail.html', {'question': question})
    question = get_object_or_404(Question, pk=question_id)
    return render(request, 'detail.html', {'question': question})

def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # 다시 질분을 디스플레이 해준다.
        return render(request, 'songs/detail.html', {
            'question': question,
            'error_message': "선택을 해주세요.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        #일반적인 서버사이드 리다이렉션방식이다.
        #reverse -> 'results/3'을 리버스해주어서 '3/results' = 'songs/3/results'가된다.
        return HttpResponseRedirect(reverse('songs:results', args=(question.id,)))
    
 
class ResultsView(generic.DetailView):
    model = Question
    template_name = 'results.html'
       
# def results(request, question_id):
#     question = get_object_or_404(Question, pk=question_id)
#     return render(request, 'results.html', {'question': question})


