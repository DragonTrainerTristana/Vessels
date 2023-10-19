using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.AI;

public class NavmeshMovement : MonoBehaviour
{
    //Agent 설정
    NavMeshAgent agent;

    //FieldManagement 스크립트에서 긁어옴
    public GameObject[] arrivialPos;
    int arrivalPointNum;

    // 시간변수 (디버깅할때 또는 쓸만할 때)
    private float timeData;
    int arrivialNum;
    int randomInt;

    //bool arrivalState = false;

    void Start()
    {
        // 외부 변수 긁어오기
        arrivalPointNum = GameObject.Find("SimulationManagement").GetComponent<FieldManagement>().arrivalPointNum;
        arrivialPos = GameObject.Find("SimulationManagement").GetComponent<FieldManagement>().arrivalPointArray;

        // 컴포넌트 초기화
        agent = GetComponent<NavMeshAgent>();

        // 변수 초기화
        arrivialNum = 0;
        timeData = 0;
        randomInt = Random.Range(0, arrivalPointNum);

    }

    void Update()
    {
        // 타겟
        agent.SetDestination(arrivialPos[randomInt].transform.position);

        timeData += Time.deltaTime;
        if (timeData > 3) {
            timeData = 0;
            Debug.Log(arrivialPos[randomInt].name);
        }

    }
   


    private void OnTriggerEnter(Collider col)
    {
        if (col.gameObject == arrivialPos[randomInt]) {
            Debug.Log("충돌났음");
           
            arrivialNum = randomInt;
            for (;;) {
                randomInt = Random.Range(0, arrivalPointNum);
                if (randomInt != arrivialNum) break;
            
            }
            
           
        }
    }

}
