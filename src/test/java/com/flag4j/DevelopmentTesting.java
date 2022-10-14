package com.flag4j;

import java.util.Arrays;
import java.util.Random;

class Matrix {
    double[][] entries;
    int m, n;

    public Matrix(double[][] entries) {
        this.entries = Arrays.stream(entries)
                .map(double[]::clone)
                .toArray(double[][]::new);
        this.m = this.entries.length;
        this.n = this.entries[0].length;
    }

    public Matrix(double[] entries) {
        this.entries = new double[entries.length][1];
        this.m = this.entries.length;
        this.n = this.entries[0].length;

        for(int i=0; i<entries.length; i++) {
            this.entries[i][0] = entries[i];
        }
    }

    public String toString() {
        String result = "[";

        for(int i=0; i<m; i++) {
            result += "[";
            for(int j=0; j<n; j++) {
                result += this.entries[i][j];

                if(j+1!=n) {
                    result += ", ";
                }
            }

            result += "]";
            if(i+1!=m) {
                result += "\n";
            }
        }
        return result + "]";
    }
}

class Vector extends Matrix {
    public Vector(double[] entries) {
        super(entries);
    }
}

class VectorFlat {
    double[] entries;
    int m;

    public VectorFlat(double[] entries) {
        this.entries = entries.clone();
        this.m = this.entries.length;
    }

    public String toString() {
        String result = "[";

        for(int i=0; i<m; i++) {
            result += "[" + entries[i] + "]";

            if(i+1!=m) {
                result += "\n";
            }
        }

        return result += "]";
    }
}

class Multiply {
    public static Vector MatrixVector(Matrix mat, Vector vec) {
        double[] product = new double[vec.m];

        for(int i=0; i<mat.m; i++) {
            for(int j=0; j<mat.n; j++) {
                product[j] += mat.entries[i][j]*vec.entries[j][0];
            }
        }

        return new Vector(product);
    }

    public static VectorFlat MatrixVector(Matrix mat, VectorFlat vec) {
        double[] product = new double[vec.m];

        for(int i=0; i<mat.m; i++) {
            for(int j=0; j<mat.n; j++) {
                product[j] += mat.entries[i][j]*vec.entries[j];
            }
        }

        return new VectorFlat(product);
    }


    public static Matrix matMultStandard(Matrix A, Matrix B) {
        return null;
    }


    private static void dimCheck(Matrix mat, Vector vec) {
        if(mat.n!=vec.m) {
            throw new IllegalArgumentException("Illegal dimensions for matrix-vector multiplication");
        }
    }

    private static void dimCheck(Matrix mat, VectorFlat vec) {
        if(mat.n!=vec.m) {
            throw new IllegalArgumentException("Illegal dimensions for matrix-vector multiplication");
        }
    }
}


public class DevelopmentTesting {

    final static long SEED = 42l;
    final static double MAX = 256;
    final static double MIN = -256;
    final static Random RAND = new Random(SEED);
    final static int NUM_ROWS = 20;
    final static int NUM_COLS = 20;
    final static int NUM_RUNS  = 1;

    public static double[] genRandArr(int size) {
        double[] randomArr = new double[size];

        for(int i=0; i<size; i++) {
            randomArr[i] = MIN + RAND.nextDouble()*(MAX-MIN);
        }

        return randomArr;
    }
    public static double[][] genRandArr2D(int m, int n) {
        double[][] randomArr = new double[m][n];

        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                randomArr[i][j] = MIN + RAND.nextDouble()*(MAX-MIN);
            }
        }

        return randomArr;
    }

    public static void timeMVM(double[][] mat_vals,double[] vec_vals) {
        Matrix A = new Matrix(mat_vals);

        final long startTimeCreate = System.currentTimeMillis();
        Vector b = new Vector(vec_vals);
        final long endTimeCreate = System.currentTimeMillis();
        final long timeCreate = endTimeCreate - startTimeCreate;

        final long startTimeMult = System.currentTimeMillis();
        for(int i=0; i<NUM_RUNS; i++) {
            Vector y = Multiply.MatrixVector(A, b);
        }
        final long endTimeMult = System.currentTimeMillis();
        final long timeMult = endTimeMult - startTimeMult;

        System.out.println("Standard Vector Results:\n--------------------------------");
        System.out.println("Time to Create Vector: " + timeCreate + " ms.");
        System.out.println("Time to Multiply: " + timeMult + " ms.");
        System.out.println("Total Execution time: " + (timeCreate+timeMult) + " ms.");
    }

    public static void timeMVMFlat(double[][] mat_vals, double[] vec_vals) {
        Matrix A = new Matrix(mat_vals);

        final long startTimeCreate = System.currentTimeMillis();
        VectorFlat b = new VectorFlat(vec_vals);
        final long endTimeCreate = System.currentTimeMillis();
        final long timeCreate = endTimeCreate - startTimeCreate;

        final long startTimeMult = System.currentTimeMillis();
        for(int i=0; i<NUM_RUNS; i++) {
            VectorFlat y = Multiply.MatrixVector(A, b);
        }
        final long endTimeMult = System.currentTimeMillis();
        final long timeMult = endTimeMult - startTimeMult;

        System.out.println("Flat Vector Results:\n--------------------------------");
        System.out.println("Time to Create Vector: " + timeCreate + " ms.");
        System.out.println("Time to Multiply: " + timeMult + " ms.");
        System.out.println("Total Execution time: " + (timeCreate+timeMult) + " ms.");
    }

    public static void timeFillLoop(double fillValue, int size) {
        double[][] arr = new double[size][size];

        final long startTimeMult = System.currentTimeMillis();
        for(int i=0; i<size; i++) {
            for(int j=0; j<size; j++) {
                arr[i][j] = fillValue;
            }
        }
        final long endTimeMult = System.currentTimeMillis();
        final long timeTot = endTimeMult - startTimeMult;

        System.out.println("Fill with loop:\n--------------------------------");
        System.out.println("Time to fill: " + timeTot + " ms.");
    }
    public static void timeFill(double fillValue, int size) {
        double[][] arr = new double[size][size];
        double[] row = new double[size];

        final long startTimeMult = System.currentTimeMillis();
        Arrays.fill(row, fillValue);
        Arrays.fill(arr, row);
        final long endTimeMult = System.currentTimeMillis();
        final long timeTot = endTimeMult - startTimeMult;

        System.out.println("Fill:\n--------------------------------");
        System.out.println("Time to fill: " + timeTot + " ms.");
    }

    public static void main(String[] args) {
        double[][] a_vals = genRandArr2D(NUM_ROWS, NUM_COLS);
        double[] b_vals = genRandArr(NUM_ROWS);

        double[][] aa = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[][] bb = {{4, 5}, {1, 9}, {6, 7}};


//        timeMVM(a_vals, b_vals);
//        System.out.println("\n\n");
//        timeMVMFlat(a_vals, b_vals);

        double fillValue = 10;
        int size = 30000;
        timeFillLoop(fillValue, size);
        System.out.println("\n\n");
        timeFill(fillValue, size);
    }
}
