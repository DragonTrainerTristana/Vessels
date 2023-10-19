using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//ml-agent용

using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class FieldManagement : MonoBehaviour
{
    // agent관리
    public GameObject[] agents;
    public GameObject agentPrefabs;
    public Transform boatArray; // emptyGameobject -> parent로 정리

    // 여기서 Instantiate 해야함
    public GameObject[] spawnPointArray;
    public int spawnPointNum;
    public int vesselNum;
    private int randomInt;
    private int num;

    // NavmeshMovement에서 긁어감
    public GameObject[] arrivalPointArray;
    public int arrivalPointNum;

    

    void Start()
    {
        // 초기화
        agents = new GameObject[vesselNum];
        //Debug.Log(vesselNum);

        for (int i = 0; i < spawnPointNum; i++) {
            // 랜덤 생성
            randomInt = Random.Range(0, spawnPointNum);
            //Debug.Log(randomInt);
            //Debug.Log(spawnPointArray[randomInt].name);
            agents[i] = Instantiate(agentPrefabs, spawnPointArray[randomInt].transform.position, Quaternion.identity);
            agents[i].transform.SetParent(boatArray,false);
        }


    }

    // Update is called once per frame
    void Update()
    {



        
    }
}
