using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CustomField : MonoBehaviour
{

    public int numSpawnPoints = 1;
    public int numWayPoints = 1;
    public int numVesselAgents = 1;

    // Prefabs
    public GameObject prefabVessel;
    public Transform parentVessel;

    public GameObject[] vesselAgents;
    public Transform[] spawnPoints;
    public Transform[] wayPoints;

    // ���� ���ص� �Ǵ� �ֵ�
    private int randomSpawn;
    private int randomWay;

    private bool[] aliveAgent;


    void Start()
    {
        vesselAgents = new GameObject[numVesselAgents];
        aliveAgent = new bool[numVesselAgents];
        //spawnPoints = new Transform[numSpawnPoints];
        //wayPoints = new Transform[numWayPoints];

        // Spawn Each Agents;
        for (int i = 0; i < numVesselAgents; i++) {
            // ���� 
            randomSpawn = Random.Range(0, numSpawnPoints);
            randomWay = Random.Range(0, numWayPoints);

            // �Ҵ�
            vesselAgents[i] = Instantiate(prefabVessel, spawnPoints[randomSpawn].position, Quaternion.identity);
            vesselAgents[i].GetComponent<CustomAgentMovement>().myWayPoint = wayPoints[randomWay];
            vesselAgents[i].transform.parent = parentVessel;
            aliveAgent[i] = true; 
        } 
    }

    void Update()
    {
        //OnEpisodeBegin ȣ��� ��
        
    }
}
