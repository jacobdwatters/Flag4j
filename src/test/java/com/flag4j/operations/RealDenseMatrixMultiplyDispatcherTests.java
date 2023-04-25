package com.flag4j.operations;

import com.flag4j.Shape;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static com.flag4j.operations.RealDenseMatrixMultiplyDispatcher.AlgorithmNames.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

class RealDenseMatrixMultiplyDispatcherTests {

    static int[] rowList;
    static int[] colList;
    Shape shape;
    static RealDenseMatrixMultiplyDispatcher.AlgorithmNames[] algos, algosT;

    @BeforeAll
    static void setup() {
        rowList = new int[]{5, 25, 50, 150, 250, 500, 1024, 2048, 4096};
        colList = new int[]{5, 25, 50, 150, 250, 500, 1024, 2048, 4096};
        algos = new RealDenseMatrixMultiplyDispatcher.AlgorithmNames[]{
                REORDERED, REORDERED, REORDERED, REORDERED, REORDERED, REORDERED, REORDERED, REORDERED, REORDERED,
                REORDERED, REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_STANDARD, CONCURRENT_STANDARD, CONCURRENT_STANDARD,
                REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_STANDARD, CONCURRENT_STANDARD, CONCURRENT_STANDARD,
                CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED,
                CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED,
                CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED,
                CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED,
                CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED,
                CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_REORDERED, CONCURRENT_BLOCKED_REORDERED
        };
        algosT = new RealDenseMatrixMultiplyDispatcher.AlgorithmNames[]{
                MULT_T, MULT_T, MULT_T_BLOCKED, MULT_T_CONCURRENT,
                MULT_T_CONCURRENT, MULT_T_CONCURRENT, MULT_T_CONCURRENT,
                MULT_T_BLOCKED_CONCURRENT, MULT_T_BLOCKED_CONCURRENT
        };
    }


    @Test
    void selectAlgorithmTests() {
        int count = 0;

        for(int k : rowList) {
            for(int i : colList) {
                shape = new Shape(k, i);
                assertEquals(
                        algos[count++],
                        RealDenseMatrixMultiplyDispatcher.selectAlgorithm(shape, shape)
                );
            }
        }
    }


    @Test
    void selectAlgorithmTransposeTests() {
        int count = 0;

        for(int j : rowList) {
            shape = new Shape(j, j);
            assertEquals(algosT[count++], RealDenseMatrixMultiplyDispatcher.selectAlgorithmTranspose(shape));
        }
    }
}
