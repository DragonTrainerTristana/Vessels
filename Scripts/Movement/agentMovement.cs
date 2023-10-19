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
    //Agent ����
    NavMeshAgent agent;

    //FieldManagement ��ũ��Ʈ���� �ܾ��
    public GameObject[] arrivialPos;
    int arrivalPointNum;

    // �ð����� (������Ҷ� �Ǵ� ������ ��)
    private float timeData;
    int arrivialNum;
    int randomInt;

    //bool arrivalState = false;


    // Start is called before the first frame update
    public override void Initialize()
    {
        // �ܺ� ���� �ܾ����
        arrivalPointNum = GameObject.Find("SimulationManagement").GetComponent<FieldManagement>().arrivalPointNum;
        arrivialPos = GameObject.Find("SimulationManagement").GetComponent<FieldManagement>().arrivalPointArray;

        // ������Ʈ �ʱ�ȭ
        agent = GetComponent<NavMeshAgent>();

        // ���� �ʱ�ȭ
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
        //rayComponent(�Ķ����)�� ������ RayOutput�� rayOutputs������ ����
        var rayOutputs = RayPerceptionSensor
                .Perceive(rayComponent.GetRayPerceptionInput())
                .RayOutputs;
        //������ ��ü(����)�� �ִٸ�
        if (rayOutputs != null)
        {	//������ �޸� Ray�� ������ ������ RayOutputs�� �迭
            var outputLegnth = RayPerceptionSensor
                    .Perceive(rayComponent.GetRayPerceptionInput())
                    .RayOutputs
                    .Length;

            for (int i = 0; i < outputLegnth; i++)
            {	//������ Ray�� �浹��(������) ��ü�� �ִ� ���
                GameObject goHit = rayOutputs[i].HitGameObject;
                if (goHit != null)
                {	// �浹�� ��ü������ �Ÿ� ���
                    var rayDirection = rayOutputs[i].EndPositionWorld - rayOutputs[i].StartPositionWorld;
                    var scaledRayLength = rayDirection.magnitude;
                    float rayHitDistance = rayOutputs[i].HitFraction * scaledRayLength;
                    // ��ֹ��� ���� �Ÿ� �̳��� ������ -1��
                    if ((goHit.tag == "ship" || goHit.tag == "Land") && rayHitDistance < 2.4f)
                    {
                        Debug.Log("boom!!!"); // �賢��, �Ǵ� ���� �浹
                        AddReward(-0.2f); // �浹 �ɶ�����, -0.2�� �ֱ�
                    }
                }
            }
        }
    }

    private void OnTriggerEnter(Collider col)
    {
        if (col.gameObject == arrivialPos[randomInt])
        {
            // ����
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
