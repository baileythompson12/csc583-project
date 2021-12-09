import org.apache.lucene.analysis.standard.StandardAnalyzer;

import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;

import org.apache.lucene.store.Directory;
import org.tartarus.snowball.ext.PorterStemmer;
import org.apache.lucene.store.ByteBuffersDirectory;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

import org.apache.lucene.search.similarities.BooleanSimilarity;
import org.apache.lucene.search.similarities.Similarity;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;

import org.apache.lucene.queryparser.classic.QueryParser;


import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.simple.*;
import edu.stanford.nlp.util.StringUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Properties;
import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.deeplearning4j.*;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.*;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.*;
import org.ejml.*;
import org.datavec.*;

class IndexEngine {
	String inputFilePath = "";
	boolean indexExists = false;
	
	Similarity similarity = new BooleanSimilarity(); //change similarity to Boolean Similarity
	
    StandardAnalyzer analyzer = new StandardAnalyzer();
    Directory index = new ByteBuffersDirectory();
    IndexWriterConfig config = new IndexWriterConfig(analyzer);
    
	IndexWriter documentWriter;
	  
	  public IndexEngine(String inputFile) throws IOException {
		  inputFilePath = inputFile;
		  config.setSimilarity(similarity);
		  documentWriter = new IndexWriter(index, config);
		  //buildIndex();
		  buildNeuralIndex();
		  
	  }
	  
	  public void buildNeuralIndex() throws IOException {
		  //Use Word2Vec to process Wikipedia pages
		  File gFile = new File("src/resources/glove.840B.300d.10f.txt.gz");
		  WordVectors wordVectors = WordVectorSerializer.readWord2VecModel("src/resources/glove.840B.300d.10f.txt");
		  System.out.println(wordVectors);
		  Word2Vec vec = new Word2Vec();
		  
		  Collection<String> lst = wordVectors.wordsNearest("day", 10);
		  System.out.println(lst);
		  
		  File folder = new File(inputFilePath);
		  File[] listOfFiles = folder.listFiles();
		  
		  
		  int fileNum = 1;
		  for (int i = 0; i < 2; i++) { //File file : listOfFiles
			  File file = listOfFiles[i];
			  System.out.println("Indexing file " + fileNum + "/" + listOfFiles.length);
			  
			  
			  //am I accidentally building the vector again?
			  SentenceIterator iter = new LineSentenceIterator(file);
			  iter.setPreProcessor(new SentencePreProcessor() { //pre process string to turn it to lower case
				  @Override
				  public String preProcess(String sentence) {
					  return sentence.toLowerCase();
				  }
			  });
			  TokenizerFactory t = new DefaultTokenizerFactory();
			  t.setTokenPreProcessor(new CommonPreprocessor());
			  
			  Word2Vec vec = new Word2Vec.Builder()
					  .minWordFrequency(5)
					  .layerSize(100)
					  .seed(42)
					  .windowSize(5)
					  .iterate(iter)
					  .tokenizerFactory(t)
					  .build();
			  
			  vec.fit();
			  
			  
			  
			  fileNum++;
		  }
		  indexExists = true;
		  documentWriter.close();
	  }
	  
	  public void buildIndex() throws IOException {
		  //Open Wiki files and turn them into docs
		  File folder = new File(inputFilePath);
		  File[] listOfFiles = folder.listFiles();
		  
		  
		  int fileNum = 1;
		  for (File file : listOfFiles) {
			  System.out.println("Indexing file " + fileNum + "/" + listOfFiles.length);
			  try(Scanner inputScanner = new Scanner(file)) {
				  String n = inputScanner.nextLine();
				  while (inputScanner.hasNextLine()) {
					 if (n.startsWith("[[")) {
						  String docid = n.substring(2, n.length() - 2);
						  
						  StringBuffer text = new StringBuffer();
						  boolean newLine = false;
						  while (!newLine && inputScanner.hasNextLine()) {
							  String line = inputScanner.nextLine();
							  if(line.startsWith("[[") && !line.startsWith("[[File")) {
								  newLine = true;
								  n = line;
							  }
							  else text.append(line + " \t");
						  }

						  //normalize text content (lemmas) with Stanford Core NLP
						  /*StringBuffer normalText = new StringBuffer();
						  Sentence sentence = new Sentence(text.toString());
						  for(String word : sentence.lemmas()) {
							  normalText.append(word + " ");
						  }*/
						  
						  
						  //normalize text content (stemming) with Lucene
						  PorterStemmer stem = new PorterStemmer();
						  stem.setCurrent(text.toString()); //change this to normalText.toString() if you uncomment the Core NLP portion
						  stem.stem();
						  String normalText = stem.getCurrent();
						  
						  
						  //System.out.println(docid + " : " + normalText.toString());
						  addDoc(documentWriter, normalText, docid);
					  }
				  }
				  inputScanner.close();
			  }
			  fileNum++;
		  }
		  indexExists = true;
		  documentWriter.close();
		  
	  }
	  
	  public void addDoc(IndexWriter w, String text, String document) throws IOException {
		  Document doc = new Document();
		  doc.add(new TextField("title", text, Field.Store.YES));
		  doc.add(new StringField("docid", document, Field.Store.YES));
		  w.addDocument(doc);
	  }
	  
	  public void runQuery(String queryFile) throws IOException, org.apache.lucene.queryparser.classic.ParseException {
		  if(!indexExists) buildIndex();
		  
		  System.out.println();
		  
		  File file = new File(queryFile);
		  int correctAnswers = 0;
		  
		  try(Scanner inputScanner = new Scanner(file)) {
			  while(inputScanner.hasNextLine()) {
				  //build query using category and clue
				  String query = "";
				  String n = inputScanner.nextLine();
				  query += n + " ";
				  n = inputScanner.nextLine();
				  query += n;
				  
				  //Make sure query doesn't have special characters that can confuse Lucene search
				  query = query.replaceAll("[^a-zA-Z0-9]", " ");  

				  // Core NLP pipeline for dependency parsing
				  Properties props = new Properties();
				  props.setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse"); //tokenize and lemmatize because we also did this to our Wiki pages
				  props.setProperty("coref.algorithm", "neural");
				  StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

				  //use dependency parsing on document
				  CoreDocument doc = new CoreDocument(query);
				  pipeline.annotate(doc);
				  CoreSentence sentence = doc.sentences().get(0);
				  SemanticGraph dependencyParse = sentence.dependencyParse();
				  String dotFormat = dependencyParse.toDotFormat();
				  System.out.println(dotFormat);
				  
				  //Lucene search query in documents 
				  String ans = "";
				  //System.out.println(query);
				  Query q = new QueryParser("title", analyzer).parse(query);
				  
				  int hitsPerPage = 1;
				  IndexReader reader = DirectoryReader.open(index);
			      IndexSearcher searcher = new IndexSearcher(reader);
			      TopDocs docs = searcher.search(q, hitsPerPage);
			      ScoreDoc[] hits = docs.scoreDocs;
			      //System.out.println(hits);
			      Document d = searcher.doc(hits[0].doc);
			      ans = d.get("docid");
			      
			      reader.close();
				  System.out.println("SEARCH ANSWER: " + ans);
				  
				  //Compare with actual answer
				  n = inputScanner.nextLine();
				  System.out.println("ACTUAL ANSWER: " + n);
				  System.out.println();
				  if (n.contains(ans)) correctAnswers++;
				  
				  if (inputScanner.hasNextLine()) n = inputScanner.nextLine();
				  
			  }
			  System.out.println(correctAnswers + "/" + 100);
			  inputScanner.close();
		  }
		  
	  }
	  
	  public static void main(String[] args) {
	        try {
	            System.out.println("******** Welcome to the Project! ********");
	            
	            String folderName = "src/resources/wiki-subset-20140602/";
	            String queryFile = "src/resources/questions.txt";
	            
	            IndexEngine objIndexEngine = new IndexEngine(folderName);
	            objIndexEngine.runQuery(queryFile);
	        }
	        catch (Exception ex) {
	            System.out.println(ex.getMessage());
	        }
	  }
	
}
