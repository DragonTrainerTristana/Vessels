using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FieldManagement : MonoBehaviour
{
    // 2023 - 10 -13 기준 개발기록
    /*
     1) FieldManagement 스크립트에서 모든 프레임 시간을 관리해야함
         
    
     2) 일단 정렬 최적화 기법이랑 다 적용시켜보면서,  
     
     
     */

    // Agent
    public GameObject prefabAgent;
    public GameObject[] agentsArray;
    public int agentNum;

    // Ocean_current data (csvRead 함수 활용)
    // 기존 좌표 -> 유니티 좌표 scaling 작업 필수
    
    bool interactionState = true; // agent script 접근 state
    
    //  



    // Start is called before the first frame update
    void Start()
    {
        agentsArray = new GameObject[agentNum];


        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    // extract data
    void csvRead()
    {


    }

}
