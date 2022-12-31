/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

import com.flag4j.Matrix;
import com.flag4j.operations.dense.real.RealMatrixMultiplication;
import com.flag4j.util.RandomTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TestingMatMult {

    static RandomTensor rng = new RandomTensor();
    static String[] algorithmNames = {"Shape", "ijk", "ikj", "ijk Blocked", "ikj Blocked",
            "ijk MT", "ikj MT", "ijk Blocked MT", "ikj Blocked MT"};
    static final String header = "%10s | %14s | %14s | %14s | %14s | %14s | %14s | %14s | %14s";
    static final String rowBase = "%10s | %14.2f | %14.2f | %14.2f | %14.2f | %14.2f | %14.2f | %14.2f | %14.2f";

    public static void runFlag4jAlgos(Matrix A, Matrix B) {
        long startTime, endTime;
        List<Double> runTimes = new ArrayList<>();

        // ---------------------- Sequential Algorithms ----------------------
        startTime = System.nanoTime();
        RealMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
        endTime = System.nanoTime();
        runTimes.add((endTime-startTime)*1.0e-6);

        startTime = System.nanoTime();
        RealMatrixMultiplication.standardReordered(A.entries, A.shape, B.entries, B.shape);
        endTime = System.nanoTime();
        runTimes.add((endTime-startTime)*1.0e-6);

        startTime = System.nanoTime();
        RealMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
        endTime = System.nanoTime();
        runTimes.add((endTime-startTime)*1.0e-6);

        startTime = System.nanoTime();
        RealMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
        endTime = System.nanoTime();
        runTimes.add((endTime-startTime)*1.0e-6);

        // --------------------- Concurrent Algorithms ---------------------
        startTime = System.nanoTime();
        RealMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
        endTime = System.nanoTime();
        runTimes.add((endTime-startTime)*1.0e-6);

        startTime = System.nanoTime();
        RealMatrixMultiplication.concurrentStandardReordered(A.entries, A.shape, B.entries, B.shape);
        endTime = System.nanoTime();
        runTimes.add((endTime-startTime)*1.0e-6);

        startTime = System.nanoTime();
        RealMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
        endTime = System.nanoTime();
        runTimes.add((endTime-startTime)*1.0e-6);

        startTime = System.nanoTime();
        RealMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
        endTime = System.nanoTime();
        runTimes.add((endTime-startTime)*1.0e-6);

        Object[] row = new Object[runTimes.size()+1];
        row[0] = A.shape.toString();
        for(int i=1; i<row.length; i++) {
            row[i] = runTimes.get(i-1);
        }

        System.out.println(String.format(rowBase, row));
    }

    public static void main(String[] args) {
        int[] sizeList = {5, 10, 32, 64, 100, 500, 1024, 2048};
        int numRows;
        int numCols;

        System.out.println("Flag4j Square Matrix-Matrix Multiply Benchmarks (Runtimes in ms):");
        System.out.println("System Info: OS-Widows; CPU-Intel i7 12700k 3.6 GHz; Cores-12; Logical Processors-20; RAM-32 GB.\n");

        System.out.println(String.format(header, algorithmNames));
        System.out.println("----------------------------------------------------------------------" +
                "----------------------------------------------------------------------------");

        for(int size : sizeList) {
            numRows = size;
            numCols = numRows;
            Matrix A = rng.getRandomMatrix(numRows, numCols);
            Matrix B = rng.getRandomMatrix(numRows, numCols);
            runFlag4jAlgos(A, B);
        }
    }
}
