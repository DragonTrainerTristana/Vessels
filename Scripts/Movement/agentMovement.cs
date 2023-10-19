using System;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.AI;

public class agentMovement : Agent
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


    // Start is called before the first frame update
    public override void Initialize()
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

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(this.transform.localPosition); // 3
    }

    public void Update()
    {
        agent.SetDestination(arrivialPos[randomInt].transform.position);

        timeData += Time.deltaTime;
        if (timeData > 3)
        {
            timeData = 0;
            Debug.Log(arrivialPos[randomInt].name);
        }
    }

    //public void MoveAgent(ActionSegment<int> act)
    //{
    //    agent.SetDestination(arrivialPos[randomInt].transform.position);

    //    timeData += Time.deltaTime;
    //    if (timeData > 3)
    //    {
    //        timeData = 0;
    //        Debug.Log(arrivialPos[randomInt].name);
    //    }

    //}

    private void RayCastInfo(RayPerceptionSensorComponent3D rayComponent)
    {
        //rayComponent(파라미터)가 만들어내는 RayOutput을 rayOutputs변수에 저장
        var rayOutputs = RayPerceptionSensor
                .Perceive(rayComponent.GetRayPerceptionInput())
                .RayOutputs;
        //감지한 물체(정보)가 있다면
        if (rayOutputs != null)
        {	//센서에 달린 Ray가 여러개 있으니 RayOutputs는 배열
            var outputLegnth = RayPerceptionSensor
                    .Perceive(rayComponent.GetRayPerceptionInput())
                    .RayOutputs
                    .Length;

            for (int i = 0; i < outputLegnth; i++)
            {	//센서의 Ray에 충돌한(감지된) 물체가 있는 경우
                GameObject goHit = rayOutputs[i].HitGameObject;
                if (goHit != null)
                {	// 충돌한 물체까지의 거리 계산
                    var rayDirection = rayOutputs[i].EndPositionWorld - rayOutputs[i].StartPositionWorld;
                    var scaledRayLength = rayDirection.magnitude;
                    float rayHitDistance = rayOutputs[i].HitFraction * scaledRayLength;
                    // 장애물이 일정 거리 이내로 들어오면 -1점
                    if ((goHit.tag == "ship" || goHit.tag == "Land") && rayHitDistance < 2.4f)
                    {
                        Debug.Log("boom!!!"); // 배끼리, 또는 땅과 충돌
                        AddReward(-0.2f); // 충돌 될때마다, -0.2점 주기
                    }
                }
            }
        }
    }

    private void OnTriggerEnter(Collider col)
    {
        if (col.gameObject == arrivialPos[randomInt])
        {
            // 도착
            AddReward(1f);

            arrivialNum = randomInt;
            for (; ; )
            {
                randomInt = Random.Range(0, arrivalPointNum);
                if (randomInt != arrivialNum) break;

            }


        }
    }

}
