#pragma once

/*

Threads class

 - Runs a set of threads on a given function

 --------------------------------------------------------------------------------------


 int ThreadWork(Threads::Thread* thread){				//Runs with multiple threads

	int Arg = *((int*)thread->Data);

	printf("Thread Id = %i with argument %i\n", thread->Id, Arg);

	thread->SyncThreads();								//Synchronise threads

	printf("Finished (%i)\n", thread->Id);

	return 0;
 }


 int main(){

	Threads Worker;

	int Arg = 3;									//Data passed to thread

	Worker.RunThreadsAsync(4, ThreadWork, &Arg);	//Runs 'ThreadWork' with 4 threads; Async function returns immediately

	Worker.WaitFinish();							//Wait for all threads to finish work

	return 0;
}

--------------------------------------------------------------------------------------

//Methods

bool Threads::RunThreads( int nThreads, int (*Addr)(Threads::Thread*) );
bool Threads::RunThreads( int nThreads, int (*Addr)(Threads::Thread*), void* Data );

bool Threads::RunThreadsAsync( int nThreads, int (*Addr)(Threads::Thread*) );
bool Threads::RunThreadsAsync( int nThreads, int (*Addr)(Threads::Thread*), void* Data );

bool Threads::IsRunning();

void Threads::WaitFinish();

*/

#ifdef _WIN32

#include <windows.h>
#include <process.h>

#else	//Linux

#include <pthread.h>
#define _snprintf snprintf

#ifndef pthreadWin	//Define Windows variable names in terms of Linux
#define pthreadWin

#define CRITICAL_SECTION pthread_mutex_t
#define CONDITION_VARIABLE pthread_cond_t
#define HANDLE pthread_t

#define INFINITE 0
typedef unsigned long DWORD;

void InitializeCriticalSection(CRITICAL_SECTION* cs){
	pthread_mutex_init(cs, 0);
}
void DeleteCriticalSection(CRITICAL_SECTION* cs){
	pthread_mutex_destroy(cs);
}
void EnterCriticalSection(CRITICAL_SECTION* cs){
	pthread_mutex_lock(cs);
}
void LeaveCriticalSection(CRITICAL_SECTION* cs){
	pthread_mutex_unlock(cs);
}
void InitializeConditionVariable(CONDITION_VARIABLE* cv){
	pthread_cond_init(cv, 0);
}
void WakeAllConditionVariable(CONDITION_VARIABLE* cv){
	pthread_cond_broadcast(cv);
}
void SleepConditionVariableCS(CONDITION_VARIABLE* cv, CRITICAL_SECTION* cs, DWORD ms){
	pthread_cond_wait(cv, cs);
}
void WaitForSingleObject(HANDLE h, DWORD ms){
	pthread_join(h, 0);
}

#endif

#endif

class Mutex{

private:
	CRITICAL_SECTION _Mutex;		//Mutex object for shared-memory thread synchronisation

public:
	Mutex(){
		InitializeCriticalSection(&_Mutex);
	}

	void Lock(bool lock){

		if(lock){
			EnterCriticalSection(&_Mutex);
		}else{
			LeaveCriticalSection(&_Mutex);
		}

	}

	~Mutex(){
		DeleteCriticalSection(&_Mutex);
	}

};

class Threads{

public:
	struct Thread{	//Passed to thread argument

		int Id;
		int nThreads;
		Threads* ThreadsClass;
		void* Data;

		void SyncThreads(){

			int nActive;
			SyncThreads(&nActive);

		}

		void SyncThreads(int* nActive){

			ThreadsClass->Sync(Id, nActive);

		}

	};

private:
	struct ThreadData : Thread{

		Mutex* Lock;

		bool Running;	//Thread currently active
		int RetCode;	//Thread return value

		HANDLE tHandle;	//Thread handle

		ThreadData(){
			Lock = new Mutex();
		}
		~ThreadData(){
			delete Lock;
		}
	};

	//Thread info

	int nThreads;					//Number of threads running
	ThreadData* ThreadInfo;			//Data for each thread
	int (*tAddr)(Threads::Thread*);	//Thread function

	//Running status

	CRITICAL_SECTION StatusLock;
	CONDITION_VARIABLE StatusCheck;

	bool Running;			//Running

	void SetRunning(bool running){

		EnterCriticalSection(&StatusLock);

		Running = running;

		LeaveCriticalSection(&StatusLock);

		WakeAllConditionVariable(&StatusCheck);		//Inform waiting threads of status change
	}

	//Synchronisation

	CRITICAL_SECTION CriticalSection;		//Mutex object for shared-memory thread synchronisation
	CONDITION_VARIABLE ThreadsContinue[2];

	int nThreadsWait[2];
	int nWaitInd;

	int nActive;							//Number of active threads

	void Sync(int tId, int* _nActive){

		EnterCriticalSection(&CriticalSection);

		int i = nWaitInd;

		nThreadsWait[i]++;

		if(nThreadsWait[i] >= nActive){				//Ready to continue

			nWaitInd = (i==0) ? 1 : 0;				//Swap counter
			nThreadsWait[nWaitInd] = 0;

			*_nActive = nThreadsWait[i];			//Number of still active threads

			LeaveCriticalSection(&CriticalSection);

			WakeAllConditionVariable(&ThreadsContinue[i]);

		}else{										//Wait for continue condition

			while(nThreadsWait[i] < nActive)
				SleepConditionVariableCS(&ThreadsContinue[i], &CriticalSection, INFINITE);

			*_nActive = nThreadsWait[i];			//Number of still active threads; NB value of 'nActive' may have changed since check, use nThreadsWait[i]

			LeaveCriticalSection(&CriticalSection);

		}

	}

	void ThreadInactive(int tId){

		EnterCriticalSection(&CriticalSection);			//Lock sync variables

		nActive--;										//Reduce number of active threads

		int i = nWaitInd;

		bool lastSync = (nThreadsWait[i] >= nActive);	//This thread would have been last to sync

		if(lastSync){
			nWaitInd = (i==0) ? 1 : 0;					//Swap counter
			nThreadsWait[nWaitInd] = 0;					//Reset
		}

		LeaveCriticalSection(&CriticalSection);

		if(lastSync)
			WakeAllConditionVariable(&ThreadsContinue[i]);	//Tell waiting threads to recheck sync status

	}


	//Run

#ifdef _WIN32
	static unsigned int __stdcall ThreadRun(void* Data){
#else
	static void* ThreadRun(void* Data){
#endif

		ThreadData* tData = (ThreadData*)Data;					//Extended struct of thread data
		Thread* tArg = (Thread*)tData;							//Struct passed to thread function

			//Run function

		int ret = tData->ThreadsClass->tAddr(tArg);

			//Thread returned

		tData->Lock->Lock(true);

		tData->RetCode = ret;									//Thread return code
		tData->Running = false;									//Thread no longer running

		tData->Lock->Lock(false);


		tData->ThreadsClass->ThreadInactive(tArg->Id);			//Remove from sync threads


		if(tData->Id==0){										//Thread 0 wait for all threads to return

			for(int i=1; i<tData->nThreads; i++)
				WaitForSingleObject(tData->ThreadsClass->ThreadInfo[i].tHandle, INFINITE);

			tData->ThreadsClass->SetRunning(false);				//Set complete

		}

		//Thread 0 last to finish

		return 0;
	}

	bool _RunThreads(int n, int (*fPtr)(Threads::Thread*), void* Data, bool Async){

		if(IsRunning())
			return false;

		ClearThreads();		//Clear previous run

		nThreads = n;
		tAddr = fPtr;								//Start address
		ThreadInfo = new ThreadData[nThreads];		//Data for each thread

		SetRunning(true);
		nActive = nThreads;

		for(int i=0; i<n; i++){

			ThreadInfo[i].Id = i;
			ThreadInfo[i].nThreads = nThreads;
			ThreadInfo[i].ThreadsClass = this;
			ThreadInfo[i].Data = Data;
			ThreadInfo[i].RetCode = 0;
			ThreadInfo[i].Running = true;

		}

			//Create threads

		for(int i=n-1; i>=0; i--){		//Create thread 0 last

			if(i==0 && !Async)
				continue;

#if _WIN32
			HANDLE tid;
			tid = (HANDLE)_beginthreadex(0, 0, &Threads::ThreadRun, &ThreadInfo[i], 0, 0);	//Start thread (Windows)
#else
			pthread_t tid;
			pthread_create(&tid, 0, &Threads::ThreadRun, (void*)(&ThreadInfo[i]));			//Start thread (Linux)
#endif

			ThreadInfo[i].tHandle = tid;

		}

		if(!Async){

			ThreadRun((void*)(&ThreadInfo[0]));		//Main thread as thread 0 if not async

			ThreadInfo[0].tHandle = 0;
		}

		return true;
	}



	void ClearThreads(){		//Resets class

		if(nThreads==0)
			return;

		delete[] ThreadInfo;

		nThreads = 0;

	}

public:
	Threads(){

		nThreads = 0;
		Running = false;

		//Status variables

		InitializeCriticalSection(&StatusLock);		//Access status variables
		InitializeConditionVariable(&StatusCheck);	//Wait on changes to status

		//Thread synchronisation

		nWaitInd = 0;

		InitializeCriticalSection(&CriticalSection);

		for(int i=0; i<2; i++){
			InitializeConditionVariable(&ThreadsContinue[i]);
			nThreadsWait[i] = 0;
		}

	}

	~Threads(){

		WaitFinish();

		ClearThreads();

		DeleteCriticalSection(&StatusLock);
		DeleteCriticalSection(&CriticalSection);

	}

	bool RunThreads(int n, int (*fPtr)(Threads::Thread*)){

		return _RunThreads(n, fPtr, 0, false);

	}

	bool RunThreads(int n, int (*fPtr)(Threads::Thread*), void* Data){

		return _RunThreads(n, fPtr, Data, false);

	}

	bool RunThreadsAsync(int n, int (*fPtr)(Threads::Thread*)){

		return _RunThreads(n, fPtr, 0, true);

	}


	bool RunThreadsAsync(int n, int (*fPtr)(Threads::Thread*), void* Data){

		return _RunThreads(n, fPtr, Data, true);

	}

	//Status

	bool ThreadIsRunning(int tId){

		ThreadInfo[tId].Lock->Lock(true);

		bool ret = ThreadInfo[tId].Running;

		ThreadInfo[tId].Lock->Lock(false);

		return ret;
	}

	int ThreadReturnValue(int tId){

		ThreadInfo[tId].Lock->Lock(true);

		int ret = ThreadInfo[tId].RetCode;

		ThreadInfo[tId].Lock->Lock(false);

		return ret;
	}

	bool IsRunning(){

		EnterCriticalSection(&StatusLock);

		bool r = Running;

		LeaveCriticalSection(&StatusLock);

		return r;
	}

	void WaitFinish(){

		EnterCriticalSection(&StatusLock);

		while(Running)
			SleepConditionVariableCS(&StatusCheck, &StatusLock, INFINITE);

		LeaveCriticalSection(&StatusLock);

	}

};
