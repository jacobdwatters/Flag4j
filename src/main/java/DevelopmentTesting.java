import com.flag4j.Matrix;
import com.flag4j.concurrency.algorithms.addition.ConcurrentAddition;
import com.flag4j.util.RandomTensor;

public class DevelopmentTesting {

    public static void main(String[] args) {
        RandomTensor rand = new RandomTensor(42l);

        int numRows = 12000;
        int numCols = 12000;
        Matrix A = rand.getRandomMatrix(numRows, numCols);
        Matrix B = rand.getRandomMatrix(numRows, numCols);

        long startTime = System.currentTimeMillis();
        A.add(B);
        long endTime = System.currentTimeMillis();
        System.out.println("\nStandard Add: " + (endTime - startTime) + " ms\n");


        startTime = System.currentTimeMillis();
        ConcurrentAddition.add(A, B);
        endTime = System.currentTimeMillis();
        System.out.println("\nConcurrent Add: " + (endTime - startTime) + " ms\n");
    }
}
